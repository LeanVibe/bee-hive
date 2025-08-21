"""
Unified WebSocket Adapter for CommunicationHub

This adapter consolidates all WebSocket communication patterns including:
- Real-time bidirectional communication
- WebSocket server and client implementations
- Connection lifecycle management
- Message acknowledgment and retry logic
- Broadcasting and multicasting support
- Performance monitoring and health checks

Replaces multiple WebSocket implementations with a single, unified adapter.
"""

import asyncio
import json
import time
import uuid
import ssl
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass
from urllib.parse import urlparse

import websockets
from websockets.exceptions import ConnectionClosedError, WebSocketException
from websockets.server import WebSocketServerProtocol
from websockets.client import WebSocketClientProtocol

from .base_adapter import BaseProtocolAdapter, AdapterStatus, HealthStatus, MessageHandler
from ..protocols import (
    UnifiedMessage, MessageResult, SubscriptionResult, 
    ProtocolType, ConnectionConfig, Priority, MessageStatus
)


@dataclass
class WebSocketServerConfig:
    """Configuration for WebSocket server."""
    host: str = "localhost"
    port: int = 8765
    ssl_context: Optional[ssl.SSLContext] = None
    ping_interval: Optional[float] = 20
    ping_timeout: Optional[float] = 20
    close_timeout: Optional[float] = 10
    max_size: Optional[int] = 1_000_000
    max_queue: Optional[int] = 32
    compression: Optional[str] = "deflate"
    origins: Optional[List[str]] = None
    extra_headers: Optional[Dict[str, str]] = None


@dataclass
class WebSocketClientConfig:
    """Configuration for WebSocket client connections."""
    ping_interval: Optional[float] = 20
    ping_timeout: Optional[float] = 20
    close_timeout: Optional[float] = 10
    max_size: Optional[int] = 1_000_000
    max_queue: Optional[int] = 32
    compression: Optional[str] = "deflate"
    extra_headers: Optional[Dict[str, str]] = None
    connect_timeout: float = 30.0
    auto_reconnect: bool = True
    reconnect_delay: float = 5.0


@dataclass
class WebSocketConnection:
    """Information about a WebSocket connection."""
    connection_id: str
    websocket: Optional[websockets.WebSocketCommonProtocol]
    connection_type: str  # "server" or "client"
    remote_address: Optional[str]
    established_at: datetime
    last_activity: datetime
    message_queue: asyncio.Queue
    subscription_patterns: Set[str]
    pending_acks: Dict[str, float]  # message_id -> timestamp
    bytes_sent: int = 0
    bytes_received: int = 0
    messages_sent: int = 0
    messages_received: int = 0


class WebSocketAdapter(BaseProtocolAdapter):
    """
    Unified WebSocket adapter for real-time communication.
    
    Features:
    - Combined server and client functionality
    - Real-time bidirectional messaging
    - Message acknowledgment and delivery confirmation
    - Broadcasting and multicasting support
    - Connection lifecycle management with auto-reconnection
    - Performance monitoring and health checks
    """
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        
        # WebSocket-specific configuration
        self.server_config = WebSocketServerConfig()
        self.client_config = WebSocketClientConfig()
        
        # Server management
        self.websocket_server: Optional[websockets.WebSocketServer] = None
        self.server_running = False
        
        # Connection management
        self.connections: Dict[str, WebSocketConnection] = {}
        self.client_connections: Dict[str, WebSocketConnection] = {}
        self.server_connections: Dict[str, WebSocketConnection] = {}
        
        # Message routing
        self.subscriptions: Dict[str, Set[str]] = {}  # pattern -> connection_ids
        self.connection_handlers: Dict[str, MessageHandler] = {}
        
        # Performance tracking
        self.total_connections_established = 0
        self.total_connections_dropped = 0
        self.active_broadcasts = 0
    
    async def connect(self) -> bool:
        """Initialize WebSocket adapter (start server and prepare for client connections)."""
        try:
            # Start WebSocket server
            success = await self._start_server()
            if not success:
                return False
            
            # Start connection monitoring
            await self._start_connection_monitor()
            
            return True
            
        except Exception as e:
            await self._record_error(f"WebSocket adapter initialization failed: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Close all WebSocket connections and stop server."""
        try:
            # Close all client connections
            for connection_id in list(self.client_connections.keys()):
                await self._close_connection(connection_id)
            
            # Close all server connections
            for connection_id in list(self.server_connections.keys()):
                await self._close_connection(connection_id)
            
            # Stop WebSocket server
            if self.websocket_server:
                self.websocket_server.close()
                await self.websocket_server.wait_closed()
                self.server_running = False
            
        except Exception as e:
            await self._record_error(f"WebSocket disconnect error: {e}")
    
    async def send_message(self, message: UnifiedMessage) -> MessageResult:
        """Send message via WebSocket connection."""
        start_time = time.time()
        
        try:
            if not self.is_connected():
                return MessageResult(
                    success=False,
                    message_id=message.id,
                    error="WebSocket adapter not connected"
                )
            
            # Find target connection(s)
            target_connections = await self._find_target_connections(message)
            
            if not target_connections:
                return MessageResult(
                    success=False,
                    message_id=message.id,
                    error=f"No WebSocket connection found for destination: {message.destination}"
                )
            
            # Send to all target connections
            successful_sends = 0
            errors = []
            
            for connection_id in target_connections:
                try:
                    success = await self._send_to_connection(connection_id, message)
                    if success:
                        successful_sends += 1
                    else:
                        errors.append(f"Send failed to connection {connection_id}")
                except Exception as e:
                    errors.append(f"Connection {connection_id}: {str(e)}")
            
            # Calculate result
            latency_ms = (time.time() - start_time) * 1000
            success = successful_sends > 0
            
            if success:
                self._record_latency(latency_ms)
                self._record_message_sent(len(json.dumps(message.to_dict())))
            
            return MessageResult(
                success=success,
                message_id=message.id,
                protocol_used=ProtocolType.WEBSOCKET,
                latency_ms=latency_ms,
                metadata={
                    "target_connections": len(target_connections),
                    "successful_sends": successful_sends,
                    "errors": errors
                }
            )
            
        except Exception as e:
            await self._record_error(f"WebSocket message send failed: {e}")
            return MessageResult(
                success=False,
                message_id=message.id,
                error=str(e)
            )
    
    async def subscribe(
        self,
        pattern: str,
        handler: Callable[[UnifiedMessage], Any],
        connection_filter: Optional[str] = None,
        **kwargs
    ) -> SubscriptionResult:
        """
        Subscribe to messages on WebSocket connections.
        
        Args:
            pattern: Message pattern to match (destination or routing key)
            handler: Async callback function for messages
            connection_filter: Filter connections by type ("server", "client", or specific connection_id)
            **kwargs: Additional options
        """
        try:
            subscription_id = str(uuid.uuid4())
            message_handler = MessageHandler(subscription_id, handler, pattern)
            
            # Register handler
            self.connection_handlers[subscription_id] = message_handler
            
            # Add pattern subscription
            if pattern not in self.subscriptions:
                self.subscriptions[pattern] = set()
            
            # Apply to matching connections
            for connection_id, connection in self.connections.items():
                if self._connection_matches_filter(connection, connection_filter):
                    connection.subscription_patterns.add(pattern)
                    self.subscriptions[pattern].add(connection_id)
            
            self.metrics.active_subscriptions += 1
            
            return SubscriptionResult(
                success=True,
                subscription_id=subscription_id,
                pattern=pattern,
                metadata={
                    "connection_filter": connection_filter,
                    "matching_connections": len(self.subscriptions[pattern])
                }
            )
            
        except Exception as e:
            await self._record_error(f"WebSocket subscription failed: {e}")
            return SubscriptionResult(
                success=False,
                subscription_id="",
                pattern=pattern,
                error=str(e)
            )
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from WebSocket messages."""
        try:
            if subscription_id not in self.connection_handlers:
                return False
            
            handler = self.connection_handlers[subscription_id]
            pattern = handler.pattern
            
            # Remove from pattern subscriptions
            if pattern in self.subscriptions:
                # Remove pattern from all connections
                for connection in self.connections.values():
                    connection.subscription_patterns.discard(pattern)
                
                del self.subscriptions[pattern]
            
            # Remove handler
            del self.connection_handlers[subscription_id]
            self.metrics.active_subscriptions -= 1
            
            return True
            
        except Exception as e:
            await self._record_error(f"WebSocket unsubscribe failed: {e}")
            return False
    
    async def health_check(self) -> HealthStatus:
        """Perform WebSocket health check."""
        try:
            # Check server status
            if not self.server_running:
                return HealthStatus.UNHEALTHY
            
            # Check active connections
            active_connections = len(self.connections)
            
            # Check connection health
            unhealthy_connections = 0
            for connection in self.connections.values():
                if (connection.websocket and 
                    (connection.websocket.closed or 
                     datetime.utcnow() - connection.last_activity > timedelta(minutes=5))):
                    unhealthy_connections += 1
            
            # Determine health status
            if active_connections == 0:
                return HealthStatus.DEGRADED
            elif unhealthy_connections > active_connections * 0.5:
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.HEALTHY
                
        except Exception:
            return HealthStatus.UNHEALTHY
    
    # === SERVER MANAGEMENT ===
    
    async def _start_server(self) -> bool:
        """Start WebSocket server."""
        try:
            self.websocket_server = await websockets.serve(
                self._handle_server_connection,
                self.server_config.host,
                self.server_config.port,
                ssl=self.server_config.ssl_context,
                ping_interval=self.server_config.ping_interval,
                ping_timeout=self.server_config.ping_timeout,
                close_timeout=self.server_config.close_timeout,
                max_size=self.server_config.max_size,
                max_queue=self.server_config.max_queue,
                compression=self.server_config.compression,
                origins=self.server_config.origins,
                extra_headers=self.server_config.extra_headers
            )
            
            self.server_running = True
            
            # Record server connection info
            connection_info = self._create_connection_info(
                "websocket_server",
                host=self.server_config.host,
                port=self.server_config.port,
                ssl_enabled=self.server_config.ssl_context is not None
            )
            self.connections["websocket_server"] = connection_info
            self.metrics.connection_count = 1
            
            return True
            
        except Exception as e:
            await self._record_error(f"WebSocket server start failed: {e}")
            return False
    
    async def _handle_server_connection(
        self,
        websocket: WebSocketServerProtocol,
        path: str
    ) -> None:
        """Handle incoming WebSocket connection."""
        connection_id = str(uuid.uuid4())
        
        try:
            # Create connection info
            connection = WebSocketConnection(
                connection_id=connection_id,
                websocket=websocket,
                connection_type="server",
                remote_address=f"{websocket.remote_address[0]}:{websocket.remote_address[1]}",
                established_at=datetime.utcnow(),
                last_activity=datetime.utcnow(),
                message_queue=asyncio.Queue(maxsize=1000),
                subscription_patterns=set(),
                pending_acks={}
            )
            
            # Register connection
            self.connections[connection_id] = connection
            self.server_connections[connection_id] = connection
            self.total_connections_established += 1
            
            # Apply existing subscriptions
            for pattern in self.subscriptions:
                connection.subscription_patterns.add(pattern)
                self.subscriptions[pattern].add(connection_id)
            
            # Start message handling
            await self._handle_connection_messages(connection_id)
            
        except Exception as e:
            await self._record_error(f"Server connection handling failed: {e}")
        finally:
            await self._close_connection(connection_id)
    
    # === CLIENT MANAGEMENT ===
    
    async def connect_to_server(
        self,
        uri: str,
        connection_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Connect to external WebSocket server as client.
        
        Args:
            uri: WebSocket server URI
            connection_id: Optional connection ID (auto-generated if not provided)
            **kwargs: Additional connection parameters
            
        Returns:
            str: Connection ID of established connection
        """
        if connection_id is None:
            connection_id = str(uuid.uuid4())
        
        try:
            # Parse URI for connection info
            parsed_uri = urlparse(uri)
            
            # Establish connection
            websocket = await websockets.connect(
                uri,
                ssl=self.client_config.ssl_context if parsed_uri.scheme == "wss" else None,
                ping_interval=self.client_config.ping_interval,
                ping_timeout=self.client_config.ping_timeout,
                close_timeout=self.client_config.close_timeout,
                max_size=self.client_config.max_size,
                max_queue=self.client_config.max_queue,
                compression=self.client_config.compression,
                extra_headers=self.client_config.extra_headers,
                **kwargs
            )
            
            # Create connection info
            connection = WebSocketConnection(
                connection_id=connection_id,
                websocket=websocket,
                connection_type="client",
                remote_address=f"{parsed_uri.hostname}:{parsed_uri.port}",
                established_at=datetime.utcnow(),
                last_activity=datetime.utcnow(),
                message_queue=asyncio.Queue(maxsize=1000),
                subscription_patterns=set(),
                pending_acks={}
            )
            
            # Register connection
            self.connections[connection_id] = connection
            self.client_connections[connection_id] = connection
            self.total_connections_established += 1
            
            # Apply existing subscriptions
            for pattern in self.subscriptions:
                connection.subscription_patterns.add(pattern)
                self.subscriptions[pattern].add(connection_id)
            
            # Start message handling
            asyncio.create_task(self._handle_connection_messages(connection_id))
            
            return connection_id
            
        except Exception as e:
            await self._record_error(f"Client connection failed: {e}")
            raise e
    
    # === MESSAGE HANDLING ===
    
    async def _handle_connection_messages(self, connection_id: str) -> None:
        """Handle messages for a WebSocket connection."""
        try:
            connection = self.connections[connection_id]
            websocket = connection.websocket
            
            async for message in websocket:
                try:
                    # Update activity
                    connection.last_activity = datetime.utcnow()
                    
                    # Parse message
                    message_data = json.loads(message)
                    
                    # Handle acknowledgments
                    if message_data.get("type") == "ack":
                        self._handle_acknowledgment(connection_id, message_data.get("message_id"))
                        continue
                    
                    # Convert to UnifiedMessage
                    unified_message = UnifiedMessage.from_dict(message_data)
                    
                    # Send acknowledgment if required
                    if message_data.get("require_ack", False):
                        await self._send_acknowledgment(websocket, unified_message.id)
                    
                    # Route to handlers
                    await self._route_message_to_handlers(connection_id, unified_message)
                    
                    # Update metrics
                    connection.messages_received += 1
                    connection.bytes_received += len(message)
                    self._record_message_received(len(message))
                    
                except json.JSONDecodeError:
                    await self._record_error(f"Invalid JSON from connection {connection_id}")
                except Exception as e:
                    await self._record_error(f"Message handling error for {connection_id}: {e}")
            
        except ConnectionClosedError:
            # Normal connection closure
            pass
        except Exception as e:
            await self._record_error(f"Connection message handler failed: {e}")
    
    async def _route_message_to_handlers(
        self,
        connection_id: str,
        message: UnifiedMessage
    ) -> None:
        """Route message to appropriate handlers based on subscriptions."""
        try:
            connection = self.connections[connection_id]
            
            # Find matching handlers
            for pattern in connection.subscription_patterns:
                if self._message_matches_pattern(message, pattern):
                    for handler_id, handler in self.connection_handlers.items():
                        if handler.pattern == pattern:
                            await handler.handle_message(message)
                            break
        
        except Exception as e:
            await self._record_error(f"Message routing failed: {e}")
    
    async def _send_to_connection(
        self,
        connection_id: str,
        message: UnifiedMessage
    ) -> bool:
        """Send message to specific WebSocket connection."""
        try:
            connection = self.connections.get(connection_id)
            if not connection or not connection.websocket:
                return False
            
            # Prepare message data
            message_data = message.to_dict()
            message_data["require_ack"] = True  # Request acknowledgment
            
            # Send message
            await connection.websocket.send(json.dumps(message_data))
            
            # Track for acknowledgment
            connection.pending_acks[message.id] = time.time()
            
            # Update metrics
            message_size = len(json.dumps(message_data))
            connection.messages_sent += 1
            connection.bytes_sent += message_size
            connection.last_activity = datetime.utcnow()
            
            return True
            
        except Exception as e:
            await self._record_error(f"Send to connection {connection_id} failed: {e}")
            return False
    
    async def _send_acknowledgment(
        self,
        websocket: websockets.WebSocketCommonProtocol,
        message_id: str
    ) -> None:
        """Send acknowledgment for received message."""
        try:
            ack_data = {
                "type": "ack",
                "message_id": message_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            await websocket.send(json.dumps(ack_data))
        except Exception as e:
            await self._record_error(f"Failed to send acknowledgment: {e}")
    
    def _handle_acknowledgment(self, connection_id: str, message_id: str) -> None:
        """Handle received message acknowledgment."""
        try:
            connection = self.connections.get(connection_id)
            if connection and message_id in connection.pending_acks:
                # Calculate acknowledgment time
                sent_time = connection.pending_acks[message_id]
                ack_time = (time.time() - sent_time) * 1000  # ms
                
                # Remove from pending
                del connection.pending_acks[message_id]
                
                # Record latency
                self._record_latency(ack_time)
        
        except Exception as e:
            # Fixed: Removed 'await' from non-async function call
            self._record_error(f"Acknowledgment handling failed: {e}")
    
    # === CONNECTION MANAGEMENT ===
    
    async def _find_target_connections(self, message: UnifiedMessage) -> List[str]:
        """Find target connections for message based on destination."""
        target_connections = []
        
        # Direct destination match
        if message.destination in self.connections:
            target_connections.append(message.destination)
        
        # Pattern-based routing
        for connection_id, connection in self.connections.items():
            for pattern in connection.subscription_patterns:
                if self._message_matches_pattern(message, pattern):
                    target_connections.append(connection_id)
                    break
        
        return list(set(target_connections))  # Remove duplicates
    
    def _message_matches_pattern(self, message: UnifiedMessage, pattern: str) -> bool:
        """Check if message matches subscription pattern."""
        # Simple pattern matching (can be enhanced with regex or glob patterns)
        if pattern == "*":
            return True
        
        # Check destination
        if message.destination == pattern:
            return True
        
        # Check routing key
        if message.routing_key == pattern:
            return True
        
        # Check message type
        if message.message_type.value == pattern:
            return True
        
        return False
    
    def _connection_matches_filter(
        self,
        connection: WebSocketConnection,
        connection_filter: Optional[str]
    ) -> bool:
        """Check if connection matches filter criteria."""
        if not connection_filter:
            return True
        
        if connection_filter in ["server", "client"]:
            return connection.connection_type == connection_filter
        
        return connection.connection_id == connection_filter
    
    async def _close_connection(self, connection_id: str) -> None:
        """Close and clean up WebSocket connection."""
        try:
            connection = self.connections.get(connection_id)
            if not connection:
                return
            
            # Close WebSocket
            if connection.websocket and not connection.websocket.closed:
                await connection.websocket.close()
            
            # Remove from subscriptions
            for pattern in connection.subscription_patterns:
                if pattern in self.subscriptions:
                    self.subscriptions[pattern].discard(connection_id)
            
            # Remove from connection registries
            self.connections.pop(connection_id, None)
            self.server_connections.pop(connection_id, None)
            self.client_connections.pop(connection_id, None)
            
            self.total_connections_dropped += 1
            
        except Exception as e:
            await self._record_error(f"Connection close failed: {e}")
    
    async def _start_connection_monitor(self) -> None:
        """Start background connection monitoring."""
        async def connection_monitor():
            while not self._shutdown_event.is_set():
                try:
                    await asyncio.sleep(30)  # Check every 30 seconds
                    
                    # Check for inactive connections
                    inactive_connections = []
                    cutoff_time = datetime.utcnow() - timedelta(minutes=5)
                    
                    for connection_id, connection in self.connections.items():
                        if (connection.last_activity < cutoff_time or
                            (connection.websocket and connection.websocket.closed)):
                            inactive_connections.append(connection_id)
                    
                    # Clean up inactive connections
                    for connection_id in inactive_connections:
                        await self._close_connection(connection_id)
                    
                    # Update metrics
                    self.metrics.connection_count = len(self.connections)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    await self._record_error(f"Connection monitor error: {e}")
        
        task = asyncio.create_task(connection_monitor())
        self._background_tasks.append(task)
    
    # === BROADCASTING ===
    
    async def broadcast_message(
        self,
        message: UnifiedMessage,
        connection_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Broadcast message to all matching connections.
        
        Args:
            message: Message to broadcast
            connection_filter: Filter connections ("server", "client", or None for all)
            
        Returns:
            Dict with broadcast results
        """
        try:
            self.active_broadcasts += 1
            start_time = time.time()
            
            # Find target connections
            target_connections = [
                conn_id for conn_id, conn in self.connections.items()
                if self._connection_matches_filter(conn, connection_filter)
            ]
            
            # Send to all targets
            successful_sends = 0
            failed_sends = 0
            
            for connection_id in target_connections:
                try:
                    success = await self._send_to_connection(connection_id, message)
                    if success:
                        successful_sends += 1
                    else:
                        failed_sends += 1
                except Exception:
                    failed_sends += 1
            
            broadcast_time = (time.time() - start_time) * 1000
            
            return {
                "success": successful_sends > 0,
                "target_connections": len(target_connections),
                "successful_sends": successful_sends,
                "failed_sends": failed_sends,
                "broadcast_time_ms": broadcast_time
            }
            
        except Exception as e:
            await self._record_error(f"Broadcast failed: {e}")
            return {"success": False, "error": str(e)}
        finally:
            self.active_broadcasts -= 1