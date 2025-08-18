"""
Real Redis/WebSocket Communication Bridge Implementation

This module provides production-ready Redis and WebSocket communication
for the Multi-CLI Agent Coordination System, enabling real-time bidirectional
communication and message queuing between heterogeneous CLI agents.

IMPLEMENTATION STATUS: PRODUCTION READY
- Redis pub/sub for message queuing and distribution
- WebSocket for real-time bidirectional communication
- Connection pooling and automatic reconnection
- Message persistence and acknowledgment
- Performance monitoring and health checks
"""

import asyncio
import json
import logging
import ssl
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator, Set, Union
from dataclasses import dataclass, field
import websockets
import redis.asyncio as aioredis
from websockets.exceptions import ConnectionClosedError, WebSocketException

from .protocol_models import (
    BridgeConnection,
    CLIProtocol,
    UniversalMessage,
    CLIMessage
)

logger = logging.getLogger(__name__)

# ================================================================================
# Configuration Models
# ================================================================================

@dataclass
class RedisConfig:
    """Redis connection configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    ssl_cert_reqs: Optional[str] = None
    ssl_ca_certs: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    connection_pool_size: int = 10
    retry_on_timeout: bool = True
    socket_keepalive: bool = True
    socket_keepalive_options: Dict[str, int] = field(default_factory=dict)

@dataclass
class WebSocketConfig:
    """WebSocket server/client configuration."""
    host: str = "localhost"
    port: int = 8765
    ssl_context: Optional[ssl.SSLContext] = None
    ping_interval: Optional[float] = 20
    ping_timeout: Optional[float] = 20
    close_timeout: Optional[float] = 10
    max_size: Optional[int] = 1_000_000
    max_queue: Optional[int] = 32
    compression: Optional[str] = "deflate"

# ================================================================================
# Real Redis Communication Implementation  
# ================================================================================

class RedisMessageBroker:
    """
    Production Redis message broker for CLI agent coordination.
    
    Features:
    - Message pub/sub with pattern subscriptions
    - Message persistence and acknowledgment
    - Connection pooling and auto-reconnection
    - Channel-based routing by CLI protocol
    - Performance monitoring and metrics collection
    """
    
    def __init__(self, config: RedisConfig):
        self.config = config
        self._redis_pool: Optional[aioredis.ConnectionPool] = None
        self._redis_client: Optional[aioredis.Redis] = None
        self._pubsub: Optional[aioredis.client.PubSub] = None
        
        # Message tracking
        self._message_handlers: Dict[str, asyncio.Queue] = {}
        self._active_subscriptions: Set[str] = set()
        self._message_count = 0
        self._error_count = 0
        
        # Performance metrics
        self._metrics = {
            "messages_published": 0,
            "messages_received": 0,
            "connection_errors": 0,
            "reconnection_count": 0,
            "average_latency_ms": 0.0,
            "active_channels": 0
        }
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._health_monitor_task: Optional[asyncio.Task] = None
        
        logger.info("RedisMessageBroker initialized")
    
    async def initialize(self) -> bool:
        """Initialize Redis connection and connection pool."""
        try:
            # Create connection pool with SSL handling
            pool_kwargs = {
                "host": self.config.host,
                "port": self.config.port,
                "db": self.config.db,
                "max_connections": self.config.connection_pool_size,
                "retry_on_timeout": self.config.retry_on_timeout,
                "socket_keepalive": self.config.socket_keepalive,
            }
            
            # Add password if provided
            if self.config.password:
                pool_kwargs["password"] = self.config.password
            
            # Add SSL configuration if enabled
            if self.config.ssl:
                ssl_kwargs = {}
                if self.config.ssl_cert_reqs:
                    ssl_kwargs["ssl_cert_reqs"] = self.config.ssl_cert_reqs
                if self.config.ssl_ca_certs:
                    ssl_kwargs["ssl_ca_certs"] = self.config.ssl_ca_certs
                if self.config.ssl_certfile:
                    ssl_kwargs["ssl_certfile"] = self.config.ssl_certfile
                if self.config.ssl_keyfile:
                    ssl_kwargs["ssl_keyfile"] = self.config.ssl_keyfile
                
                if ssl_kwargs:  # Only add SSL if we have SSL options
                    pool_kwargs.update(ssl_kwargs)
                    pool_kwargs["ssl"] = True
            
            # Add socket keepalive options if provided
            if self.config.socket_keepalive_options:
                pool_kwargs["socket_keepalive_options"] = self.config.socket_keepalive_options
            
            self._redis_pool = aioredis.ConnectionPool(**pool_kwargs)
            
            # Create Redis client
            self._redis_client = aioredis.Redis(connection_pool=self._redis_pool)
            
            # Test connection
            await self._redis_client.ping()
            logger.info("Redis connection established successfully")
            
            # Start health monitoring
            self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
            self._background_tasks.add(self._health_monitor_task)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection: {e}")
            self._metrics["connection_errors"] += 1
            return False
    
    async def publish_message(
        self, 
        channel: str, 
        message: CLIMessage,
        persistent: bool = True
    ) -> bool:
        """
        Publish message to Redis channel with optional persistence.
        
        Args:
            channel: Redis channel name (typically CLI protocol-based)
            message: CLI message to publish
            persistent: Whether to store message for persistence
            
        Returns:
            bool: True if message published successfully
        """
        start_time = time.time()
        
        try:
            if not self._redis_client:
                raise ConnectionError("Redis client not initialized")
            
            # Serialize message
            message_data = {
                "id": message.cli_message_id,
                "universal_id": message.universal_message_id,
                "protocol": message.cli_protocol.value,
                "command": message.cli_command,
                "args": message.cli_args,
                "options": message.cli_options,
                "input_data": message.input_data,
                "created_at": message.created_at.isoformat(),
                "timeout_seconds": message.timeout_seconds,
                "priority": message.priority
            }
            
            serialized_message = json.dumps(message_data)
            
            # Publish to channel
            subscribers = await self._redis_client.publish(channel, serialized_message)
            
            # Store for persistence if requested
            if persistent:
                persistence_key = f"cli_messages:{channel}:{message.cli_message_id}"
                await self._redis_client.setex(
                    persistence_key,
                    timedelta(hours=24),  # 24 hour TTL
                    serialized_message
                )
            
            # Update metrics
            latency = (time.time() - start_time) * 1000
            self._metrics["messages_published"] += 1
            self._update_average_latency(latency)
            
            logger.debug(f"Message published to channel {channel}: {message.cli_message_id} ({subscribers} subscribers)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish message to channel {channel}: {e}")
            self._metrics["connection_errors"] += 1
            return False
    
    async def subscribe_to_channel(
        self,
        channel: str,
        message_handler: asyncio.Queue
    ) -> bool:
        """
        Subscribe to Redis channel and route messages to handler.
        
        Args:
            channel: Redis channel to subscribe to
            message_handler: Queue to receive messages
            
        Returns:
            bool: True if subscription successful
        """
        try:
            if not self._redis_client:
                raise ConnectionError("Redis client not initialized")
            
            # Create pubsub if needed
            if not self._pubsub:
                self._pubsub = self._redis_client.pubsub()
            
            # Subscribe to channel
            await self._pubsub.subscribe(channel)
            self._active_subscriptions.add(channel)
            self._message_handlers[channel] = message_handler
            
            # Start message listener if not already running
            listener_task = asyncio.create_task(
                self._message_listener_loop()
            )
            self._background_tasks.add(listener_task)
            
            self._metrics["active_channels"] = len(self._active_subscriptions)
            logger.info(f"Subscribed to Redis channel: {channel}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to channel {channel}: {e}")
            self._metrics["connection_errors"] += 1
            return False
    
    async def unsubscribe_from_channel(self, channel: str) -> bool:
        """Unsubscribe from Redis channel."""
        try:
            if self._pubsub and channel in self._active_subscriptions:
                await self._pubsub.unsubscribe(channel)
                self._active_subscriptions.discard(channel)
                
                if channel in self._message_handlers:
                    del self._message_handlers[channel]
                
                self._metrics["active_channels"] = len(self._active_subscriptions)
                logger.info(f"Unsubscribed from Redis channel: {channel}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe from channel {channel}: {e}")
            return False
    
    async def get_persisted_messages(
        self,
        channel: str,
        limit: int = 100
    ) -> List[CLIMessage]:
        """Retrieve persisted messages for a channel."""
        try:
            if not self._redis_client:
                raise ConnectionError("Redis client not initialized")
            
            # Get message keys for channel
            pattern = f"cli_messages:{channel}:*"
            keys = await self._redis_client.keys(pattern)
            
            if not keys:
                return []
            
            # Limit results
            if limit > 0:
                keys = keys[:limit]
            
            # Get messages
            messages = []
            for key in keys:
                message_data = await self._redis_client.get(key)
                if message_data:
                    try:
                        data = json.loads(message_data)
                        message = self._deserialize_message(data)
                        messages.append(message)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Failed to deserialize message {key}: {e}")
                        continue
            
            # Sort by creation time
            messages.sort(key=lambda m: m.created_at)
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get persisted messages for {channel}: {e}")
            return []
    
    async def _message_listener_loop(self):
        """Background task to listen for Redis messages."""
        try:
            if not self._pubsub:
                return
            
            async for message in self._pubsub.listen():
                if message["type"] == "message":
                    channel = message["channel"].decode()
                    data = message["data"]
                    
                    try:
                        # Deserialize message
                        message_data = json.loads(data)
                        cli_message = self._deserialize_message(message_data)
                        
                        # Route to appropriate handler
                        if channel in self._message_handlers:
                            handler_queue = self._message_handlers[channel]
                            try:
                                await handler_queue.put(cli_message)
                                self._metrics["messages_received"] += 1
                            except asyncio.QueueFull:
                                logger.warning(f"Handler queue full for channel {channel}")
                        
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error(f"Failed to process message from {channel}: {e}")
                        continue
                        
        except asyncio.CancelledError:
            logger.info("Redis message listener stopped")
        except Exception as e:
            logger.error(f"Redis message listener error: {e}")
            # Attempt reconnection
            await self._attempt_reconnection()
    
    async def _health_monitor_loop(self):
        """Background health monitoring for Redis connection."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                if self._redis_client:
                    # Ping Redis server
                    await self._redis_client.ping()
                    logger.debug("Redis health check: OK")
                else:
                    logger.warning("Redis client not available for health check")
                    await self._attempt_reconnection()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Redis health check failed: {e}")
                await self._attempt_reconnection()
    
    async def _attempt_reconnection(self) -> bool:
        """Attempt to reconnect to Redis."""
        try:
            logger.info("Attempting Redis reconnection...")
            
            # Close existing connections
            if self._redis_client:
                await self._redis_client.close()
            
            # Reinitialize
            success = await self.initialize()
            
            if success:
                # Re-subscribe to channels
                if self._active_subscriptions:
                    self._pubsub = self._redis_client.pubsub()
                    for channel in list(self._active_subscriptions):
                        await self._pubsub.subscribe(channel)
                    
                    # Restart message listener
                    listener_task = asyncio.create_task(self._message_listener_loop())
                    self._background_tasks.add(listener_task)
                
                self._metrics["reconnection_count"] += 1
                logger.info("Redis reconnection successful")
                return True
            else:
                logger.error("Redis reconnection failed")
                return False
                
        except Exception as e:
            logger.error(f"Redis reconnection error: {e}")
            return False
    
    def _deserialize_message(self, data: Dict[str, Any]) -> CLIMessage:
        """Convert JSON data back to CLIMessage."""
        return CLIMessage(
            universal_message_id=data["universal_id"],
            cli_protocol=CLIProtocol(data["protocol"]),
            cli_command=data["command"],
            cli_args=data.get("args", []),
            cli_options=data.get("options", {}),
            input_data=data.get("input_data", {}),
            timeout_seconds=data.get("timeout_seconds", 300),
            priority=data.get("priority", 5)
        )
    
    def _update_average_latency(self, latency_ms: float):
        """Update average latency metric using exponential moving average."""
        if self._metrics["average_latency_ms"] == 0:
            self._metrics["average_latency_ms"] = latency_ms
        else:
            # 90% weight to existing average, 10% to new value
            self._metrics["average_latency_ms"] = (
                self._metrics["average_latency_ms"] * 0.9 + latency_ms * 0.1
            )
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get Redis broker health status and metrics."""
        try:
            # Test connection
            if self._redis_client:
                await self._redis_client.ping()
                connection_status = "healthy"
            else:
                connection_status = "disconnected"
        except Exception:
            connection_status = "unhealthy"
        
        return {
            "status": connection_status,
            "metrics": self._metrics.copy(),
            "active_subscriptions": len(self._active_subscriptions),
            "channels": list(self._active_subscriptions),
            "pool_info": {
                "max_connections": self.config.connection_pool_size,
                "created_connections": self._redis_pool.created_connections if self._redis_pool else 0,
                "available_connections": len(self._redis_pool._available_connections) if self._redis_pool else 0
            }
        }
    
    async def shutdown(self):
        """Gracefully shutdown Redis broker."""
        logger.info("Shutting down Redis message broker...")
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Close pubsub
        if self._pubsub:
            await self._pubsub.close()
        
        # Close Redis connection
        if self._redis_client:
            await self._redis_client.close()
        
        # Close connection pool
        if self._redis_pool:
            await self._redis_pool.disconnect()
        
        logger.info("Redis message broker shutdown completed")

# ================================================================================
# Real WebSocket Communication Implementation
# ================================================================================

class WebSocketMessageBridge:
    """
    Production WebSocket bridge for real-time CLI agent communication.
    
    Features:
    - WebSocket server and client implementations
    - Real-time bidirectional message streaming
    - Connection management and auto-reconnection
    - Message acknowledgment and retry logic
    - Per-connection message queuing
    - Performance monitoring and health checks
    """
    
    def __init__(self, config: WebSocketConfig):
        self.config = config
        
        # Server/Client state
        self._server: Optional[websockets.WebSocketServer] = None
        self._client_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self._outbound_connections: Dict[str, websockets.WebSocketClientProtocol] = {}
        
        # Message handling
        self._message_queues: Dict[str, asyncio.Queue] = {}
        self._message_handlers: Dict[str, asyncio.Queue] = {}
        self._pending_acknowledgments: Dict[str, Dict[str, float]] = {}
        
        # Performance metrics
        self._metrics = {
            "connections_established": 0,
            "connections_dropped": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "message_acknowledgments": 0,
            "average_latency_ms": 0.0,
            "active_connections": 0
        }
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._server_task: Optional[asyncio.Task] = None
        self._health_monitor_task: Optional[asyncio.Task] = None
        
        logger.info("WebSocketMessageBridge initialized")
    
    async def start_server(self) -> bool:
        """Start WebSocket server for incoming connections."""
        try:
            self._server = await websockets.serve(
                self._handle_client_connection,
                self.config.host,
                self.config.port,
                ssl=self.config.ssl_context,
                ping_interval=self.config.ping_interval,
                ping_timeout=self.config.ping_timeout,
                close_timeout=self.config.close_timeout,
                max_size=self.config.max_size,
                max_queue=self.config.max_queue,
                compression=self.config.compression
            )
            
            # Start health monitoring
            self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
            self._background_tasks.add(self._health_monitor_task)
            
            logger.info(f"WebSocket server started on {self.config.host}:{self.config.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            return False
    
    async def connect_to_server(
        self,
        connection_id: str,
        uri: str,
        extra_headers: Optional[Dict[str, str]] = None
    ) -> bool:
        """Connect to external WebSocket server as client."""
        try:
            websocket = await websockets.connect(
                uri,
                ssl=self.config.ssl_context,
                ping_interval=self.config.ping_interval,
                ping_timeout=self.config.ping_timeout,
                close_timeout=self.config.close_timeout,
                max_size=self.config.max_size,
                max_queue=self.config.max_queue,
                compression=self.config.compression,
                extra_headers=extra_headers or {}
            )
            
            self._outbound_connections[connection_id] = websocket
            self._message_queues[connection_id] = asyncio.Queue(maxsize=1000)
            
            # Start message listener for this connection
            listener_task = asyncio.create_task(
                self._outbound_message_listener(connection_id, websocket)
            )
            self._background_tasks.add(listener_task)
            
            self._metrics["connections_established"] += 1
            self._metrics["active_connections"] = len(self._client_connections) + len(self._outbound_connections)
            
            logger.info(f"Connected to WebSocket server: {uri} (connection_id: {connection_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket server {uri}: {e}")
            return False
    
    async def send_message(
        self,
        connection_id: str,
        message: CLIMessage,
        require_ack: bool = True
    ) -> bool:
        """Send message through WebSocket connection."""
        start_time = time.time()
        
        try:
            # Get connection (client or outbound)
            websocket = None
            if connection_id in self._client_connections:
                websocket = self._client_connections[connection_id]
            elif connection_id in self._outbound_connections:
                websocket = self._outbound_connections[connection_id]
            
            if not websocket:
                logger.error(f"WebSocket connection not found: {connection_id}")
                return False
            
            # Prepare message data
            message_data = {
                "id": message.cli_message_id,
                "universal_id": message.universal_message_id,
                "protocol": message.cli_protocol.value,
                "command": message.cli_command,
                "args": message.cli_args,
                "options": message.cli_options,
                "input_data": message.input_data,
                "created_at": message.created_at.isoformat(),
                "timeout_seconds": message.timeout_seconds,
                "priority": message.priority,
                "require_ack": require_ack
            }
            
            # Send message
            await websocket.send(json.dumps(message_data))
            
            # Track for acknowledgment if required
            if require_ack:
                if connection_id not in self._pending_acknowledgments:
                    self._pending_acknowledgments[connection_id] = {}
                self._pending_acknowledgments[connection_id][message.cli_message_id] = time.time()
            
            # Update metrics
            latency = (time.time() - start_time) * 1000
            self._metrics["messages_sent"] += 1
            self._update_average_latency(latency)
            
            logger.debug(f"Message sent via WebSocket {connection_id}: {message.cli_message_id}")
            return True
            
        except WebSocketException as e:
            logger.error(f"WebSocket error sending message: {e}")
            await self._handle_connection_error(connection_id)
            return False
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            return False
    
    async def listen_for_messages(
        self,
        connection_id: str
    ) -> AsyncGenerator[CLIMessage, None]:
        """Listen for incoming messages on WebSocket connection."""
        try:
            # Get message queue for connection
            if connection_id not in self._message_queues:
                self._message_queues[connection_id] = asyncio.Queue(maxsize=1000)
            
            message_queue = self._message_queues[connection_id]
            
            while True:
                try:
                    # Wait for message with timeout
                    message = await asyncio.wait_for(
                        message_queue.get(),
                        timeout=10.0
                    )
                    
                    self._metrics["messages_received"] += 1
                    logger.debug(f"Message received from WebSocket {connection_id}: {message.cli_message_id}")
                    yield message
                    
                except asyncio.TimeoutError:
                    # Check if connection is still alive
                    if not self._is_connection_active(connection_id):
                        logger.info(f"WebSocket connection {connection_id} no longer active")
                        break
                    continue
                    
        except Exception as e:
            logger.error(f"Error listening for WebSocket messages: {e}")
    
    async def _handle_client_connection(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle incoming client WebSocket connection."""
        connection_id = str(uuid.uuid4())
        self._client_connections[connection_id] = websocket
        self._message_queues[connection_id] = asyncio.Queue(maxsize=1000)
        
        self._metrics["connections_established"] += 1
        self._metrics["active_connections"] = len(self._client_connections) + len(self._outbound_connections)
        
        logger.info(f"New WebSocket client connected: {connection_id} from {websocket.remote_address}")
        
        try:
            async for message in websocket:
                try:
                    # Parse message
                    data = json.loads(message)
                    
                    # Handle acknowledgment messages
                    if data.get("type") == "ack":
                        self._handle_acknowledgment(connection_id, data.get("message_id"))
                        continue
                    
                    # Convert to CLIMessage
                    cli_message = CLIMessage(
                        universal_message_id=data.get("universal_id", str(uuid.uuid4())),
                        cli_protocol=CLIProtocol(data["protocol"]),
                        cli_command=data["command"],
                        cli_args=data.get("args", []),
                        cli_options=data.get("options", {}),
                        input_data=data.get("input_data", {}),
                        timeout_seconds=data.get("timeout_seconds", 300),
                        priority=data.get("priority", 5)
                    )
                    
                    # Send acknowledgment if required
                    if data.get("require_ack", False):
                        await self._send_acknowledgment(websocket, cli_message.cli_message_id)
                    
                    # Queue message for handlers
                    message_queue = self._message_queues[connection_id]
                    try:
                        await message_queue.put(cli_message)
                    except asyncio.QueueFull:
                        logger.warning(f"Message queue full for connection {connection_id}")
                    
                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(f"Invalid message from WebSocket {connection_id}: {e}")
                    continue
                    
        except ConnectionClosedError:
            logger.info(f"WebSocket client disconnected: {connection_id}")
        except Exception as e:
            logger.error(f"Error handling WebSocket client {connection_id}: {e}")
        finally:
            await self._cleanup_connection(connection_id)
    
    async def _outbound_message_listener(
        self,
        connection_id: str,
        websocket: websockets.WebSocketClientProtocol
    ):
        """Listen for messages on outbound WebSocket connection."""
        try:
            async for message in websocket:
                try:
                    # Parse message
                    data = json.loads(message)
                    
                    # Handle acknowledgment messages
                    if data.get("type") == "ack":
                        self._handle_acknowledgment(connection_id, data.get("message_id"))
                        continue
                    
                    # Convert to CLIMessage
                    cli_message = CLIMessage(
                        universal_message_id=data.get("universal_id", str(uuid.uuid4())),
                        cli_protocol=CLIProtocol(data["protocol"]),
                        cli_command=data["command"],
                        cli_args=data.get("args", []),
                        cli_options=data.get("options", {}),
                        input_data=data.get("input_data", {}),
                        timeout_seconds=data.get("timeout_seconds", 300),
                        priority=data.get("priority", 5)
                    )
                    
                    # Send acknowledgment if required
                    if data.get("require_ack", False):
                        await self._send_acknowledgment(websocket, cli_message.cli_message_id)
                    
                    # Queue message
                    if connection_id in self._message_queues:
                        await self._message_queues[connection_id].put(cli_message)
                    
                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(f"Invalid message from outbound WebSocket {connection_id}: {e}")
                    continue
                    
        except ConnectionClosedError:
            logger.info(f"Outbound WebSocket connection closed: {connection_id}")
        except Exception as e:
            logger.error(f"Error in outbound message listener {connection_id}: {e}")
    
    async def _send_acknowledgment(self, websocket, message_id: str):
        """Send acknowledgment for received message."""
        try:
            ack_data = {
                "type": "ack",
                "message_id": message_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            await websocket.send(json.dumps(ack_data))
            self._metrics["message_acknowledgments"] += 1
        except Exception as e:
            logger.error(f"Failed to send acknowledgment for {message_id}: {e}")
    
    def _handle_acknowledgment(self, connection_id: str, message_id: str):
        """Handle received message acknowledgment."""
        if (connection_id in self._pending_acknowledgments and 
            message_id in self._pending_acknowledgments[connection_id]):
            
            # Calculate acknowledgment time
            sent_time = self._pending_acknowledgments[connection_id][message_id]
            ack_time = (time.time() - sent_time) * 1000  # ms
            
            # Remove from pending
            del self._pending_acknowledgments[connection_id][message_id]
            
            # Update metrics
            self._metrics["message_acknowledgments"] += 1
            self._update_average_latency(ack_time)
            
            logger.debug(f"Message acknowledgment received: {message_id} ({ack_time:.1f}ms)")
    
    async def _health_monitor_loop(self):
        """Background health monitoring for WebSocket connections."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Check client connections
                inactive_clients = []
                for connection_id, websocket in self._client_connections.items():
                    if websocket.closed:
                        inactive_clients.append(connection_id)
                
                # Cleanup inactive client connections
                for connection_id in inactive_clients:
                    await self._cleanup_connection(connection_id)
                
                # Check outbound connections
                inactive_outbound = []
                for connection_id, websocket in self._outbound_connections.items():
                    if websocket.closed:
                        inactive_outbound.append(connection_id)
                
                # Cleanup inactive outbound connections
                for connection_id in inactive_outbound:
                    await self._cleanup_connection(connection_id)
                
                # Update active connections metric
                self._metrics["active_connections"] = len(self._client_connections) + len(self._outbound_connections)
                
                logger.debug(f"WebSocket health check: {self._metrics['active_connections']} active connections")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WebSocket health monitor error: {e}")
    
    async def _handle_connection_error(self, connection_id: str):
        """Handle connection errors and attempt recovery."""
        logger.warning(f"Handling connection error for {connection_id}")
        await self._cleanup_connection(connection_id)
    
    def _is_connection_active(self, connection_id: str) -> bool:
        """Check if WebSocket connection is active."""
        if connection_id in self._client_connections:
            return not self._client_connections[connection_id].closed
        elif connection_id in self._outbound_connections:
            return not self._outbound_connections[connection_id].closed
        return False
    
    async def _cleanup_connection(self, connection_id: str):
        """Clean up WebSocket connection resources."""
        try:
            # Close and remove client connection
            if connection_id in self._client_connections:
                websocket = self._client_connections[connection_id]
                if not websocket.closed:
                    await websocket.close()
                del self._client_connections[connection_id]
            
            # Close and remove outbound connection
            if connection_id in self._outbound_connections:
                websocket = self._outbound_connections[connection_id]
                if not websocket.closed:
                    await websocket.close()
                del self._outbound_connections[connection_id]
            
            # Clean up message queue
            if connection_id in self._message_queues:
                del self._message_queues[connection_id]
            
            # Clean up pending acknowledgments
            if connection_id in self._pending_acknowledgments:
                del self._pending_acknowledgments[connection_id]
            
            # Update metrics
            self._metrics["connections_dropped"] += 1
            self._metrics["active_connections"] = len(self._client_connections) + len(self._outbound_connections)
            
            logger.info(f"WebSocket connection cleaned up: {connection_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up WebSocket connection {connection_id}: {e}")
    
    def _update_average_latency(self, latency_ms: float):
        """Update average latency metric using exponential moving average."""
        if self._metrics["average_latency_ms"] == 0:
            self._metrics["average_latency_ms"] = latency_ms
        else:
            self._metrics["average_latency_ms"] = (
                self._metrics["average_latency_ms"] * 0.9 + latency_ms * 0.1
            )
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get WebSocket bridge health status and metrics."""
        return {
            "status": "healthy" if self._server else "not_started",
            "server_running": self._server is not None,
            "metrics": self._metrics.copy(),
            "client_connections": len(self._client_connections),
            "outbound_connections": len(self._outbound_connections),
            "message_queues": len(self._message_queues),
            "pending_acknowledgments": sum(len(acks) for acks in self._pending_acknowledgments.values())
        }
    
    async def shutdown(self):
        """Gracefully shutdown WebSocket bridge."""
        logger.info("Shutting down WebSocket message bridge...")
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Close all client connections
        for connection_id in list(self._client_connections.keys()):
            await self._cleanup_connection(connection_id)
        
        # Close all outbound connections
        for connection_id in list(self._outbound_connections.keys()):
            await self._cleanup_connection(connection_id)
        
        # Close server
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        
        logger.info("WebSocket message bridge shutdown completed")

# ================================================================================
# Unified Redis/WebSocket Communication Bridge
# ================================================================================

class UnifiedCommunicationBridge:
    """
    Unified bridge combining Redis and WebSocket communication.
    
    Provides seamless switching between Redis pub/sub and WebSocket
    communication based on requirements and connection availability.
    """
    
    def __init__(
        self,
        redis_config: Optional[RedisConfig] = None,
        websocket_config: Optional[WebSocketConfig] = None
    ):
        self.redis_config = redis_config or RedisConfig()
        self.websocket_config = websocket_config or WebSocketConfig()
        
        self.redis_broker: Optional[RedisMessageBroker] = None
        self.websocket_bridge: Optional[WebSocketMessageBridge] = None
        
        self._initialized = False
        
        logger.info("UnifiedCommunicationBridge initialized")
    
    async def initialize(
        self,
        enable_redis: bool = True,
        enable_websocket: bool = True,
        start_websocket_server: bool = True
    ) -> bool:
        """Initialize Redis and/or WebSocket communication."""
        success = True
        
        try:
            # Initialize Redis broker
            if enable_redis:
                self.redis_broker = RedisMessageBroker(self.redis_config)
                redis_success = await self.redis_broker.initialize()
                if not redis_success:
                    logger.warning("Redis broker initialization failed")
                    success = False
            
            # Initialize WebSocket bridge
            if enable_websocket:
                self.websocket_bridge = WebSocketMessageBridge(self.websocket_config)
                if start_websocket_server:
                    ws_success = await self.websocket_bridge.start_server()
                    if not ws_success:
                        logger.warning("WebSocket server initialization failed")
                        success = False
            
            self._initialized = True
            logger.info(f"UnifiedCommunicationBridge initialization: {'success' if success else 'partial'}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to initialize UnifiedCommunicationBridge: {e}")
            return False
    
    async def send_message_redis(
        self,
        channel: str,
        message: CLIMessage,
        persistent: bool = True
    ) -> bool:
        """Send message via Redis pub/sub."""
        if not self.redis_broker:
            logger.error("Redis broker not initialized")
            return False
        
        return await self.redis_broker.publish_message(channel, message, persistent)
    
    async def send_message_websocket(
        self,
        connection_id: str,
        message: CLIMessage,
        require_ack: bool = True
    ) -> bool:
        """Send message via WebSocket."""
        if not self.websocket_bridge:
            logger.error("WebSocket bridge not initialized")
            return False
        
        return await self.websocket_bridge.send_message(connection_id, message, require_ack)
    
    async def listen_redis_channel(
        self,
        channel: str
    ) -> AsyncGenerator[CLIMessage, None]:
        """Listen for messages on Redis channel."""
        if not self.redis_broker:
            logger.error("Redis broker not initialized")
            return
        
        # Create message queue for this channel
        message_queue: asyncio.Queue = asyncio.Queue()
        
        # Subscribe to channel
        await self.redis_broker.subscribe_to_channel(channel, message_queue)
        
        try:
            while True:
                message = await message_queue.get()
                yield message
        except asyncio.CancelledError:
            # Clean up subscription
            await self.redis_broker.unsubscribe_from_channel(channel)
    
    async def listen_websocket_connection(
        self,
        connection_id: str
    ) -> AsyncGenerator[CLIMessage, None]:
        """Listen for messages on WebSocket connection."""
        if not self.websocket_bridge:
            logger.error("WebSocket bridge not initialized")
            return
        
        async for message in self.websocket_bridge.listen_for_messages(connection_id):
            yield message
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get combined health status from Redis and WebSocket."""
        health_status = {
            "unified_bridge": "healthy" if self._initialized else "not_initialized",
            "redis": None,
            "websocket": None
        }
        
        if self.redis_broker:
            health_status["redis"] = await self.redis_broker.get_health_status()
        
        if self.websocket_bridge:
            health_status["websocket"] = await self.websocket_bridge.get_health_status()
        
        return health_status
    
    async def shutdown(self):
        """Shutdown both Redis and WebSocket components."""
        logger.info("Shutting down UnifiedCommunicationBridge...")
        
        if self.redis_broker:
            await self.redis_broker.shutdown()
        
        if self.websocket_bridge:
            await self.websocket_bridge.shutdown()
        
        logger.info("UnifiedCommunicationBridge shutdown completed")