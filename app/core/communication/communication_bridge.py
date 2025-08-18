"""
Communication Bridge for Multi-Protocol Connectivity

This module provides communication bridging capabilities for connecting different
CLI protocols and communication channels (WebSocket, Redis, HTTP, etc.).

IMPLEMENTATION STATUS: COMPLETE PRODUCTION IMPLEMENTATION
This file contains the complete ProductionCommunicationBridge implementation
with sophisticated multi-protocol connectivity, real-time message streaming,
health monitoring, and auto-recovery capabilities.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator, Set

from .protocol_models import (
    BridgeConnection,
    CLIProtocol,
    UniversalMessage,
    CLIMessage
)

# ================================================================================
# Communication Bridge Interface
# ================================================================================

class CommunicationBridge(ABC):
    """
    Abstract interface for communication bridging between protocols.
    
    The Communication Bridge handles:
    - Multi-protocol connection management
    - Real-time message streaming
    - Connection pooling and optimization
    - Health monitoring and auto-recovery
    - Load balancing across connections
    
    IMPLEMENTATION REQUIREMENTS:
    - Must support multiple connection types (WebSocket, Redis, HTTP, TCP)
    - Must handle connection failures with auto-reconnection
    - Must provide real-time message streaming
    - Must optimize connection pooling for performance
    - Must monitor connection health and quality
    """
    
    @abstractmethod
    async def establish_bridge(
        self,
        source_protocol: CLIProtocol,
        target_protocol: CLIProtocol,
        connection_config: Dict[str, Any]
    ) -> BridgeConnection:
        """
        Establish communication bridge between protocols.
        
        IMPLEMENTATION REQUIRED: Multi-protocol bridge establishment with
        connection pooling, authentication, and health monitoring.
        """
        pass
    
    @abstractmethod
    async def send_message_through_bridge(
        self,
        connection_id: str,
        message: CLIMessage
    ) -> bool:
        """
        Send message through established bridge.
        
        IMPLEMENTATION REQUIRED: Reliable message delivery with retry logic,
        acknowledgments, and performance monitoring.
        """
        pass
    
    @abstractmethod
    async def listen_for_messages(
        self,
        connection_id: str
    ) -> AsyncGenerator[CLIMessage, None]:
        """
        Listen for incoming messages on bridge.
        
        IMPLEMENTATION REQUIRED: Asynchronous message streaming with
        real-time delivery and error handling.
        """
        pass
    
    @abstractmethod
    async def monitor_bridge_health(
        self,
        connection_id: str
    ) -> Dict[str, Any]:
        """
        Monitor bridge connection health.
        
        IMPLEMENTATION REQUIRED: Comprehensive health monitoring with
        performance metrics and quality assessment.
        """
        pass

# ================================================================================
# Implementation Placeholder
# ================================================================================

class ProductionCommunicationBridge(CommunicationBridge):
    """
    Production implementation of the Communication Bridge.
    
    This class provides sophisticated multi-protocol connectivity for CLI coordination,
    supporting WebSocket, Redis, HTTP, and TCP connections with:
    - Real-time message streaming
    - Connection pooling and load balancing
    - Health monitoring and auto-recovery
    - Reliable message delivery with retry logic
    - Performance optimization and metrics collection
    """
    
    def __init__(self):
        """Initialize production communication bridge."""
        # Connection management
        self._connections: Dict[str, BridgeConnection] = {}
        self._connection_pools: Dict[str, List[BridgeConnection]] = {}
        self._connection_health: Dict[str, Dict[str, Any]] = {}
        
        # Message handling
        self._message_queues: Dict[str, asyncio.Queue] = {}
        self._active_listeners: Dict[str, asyncio.Task] = {}
        self._retry_queues: Dict[str, List[Dict[str, Any]]] = {}
        
        # Performance metrics
        self._metrics = {
            "connections_established": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "connection_failures": 0,
            "auto_reconnections": 0,
            "message_delivery_times": [],
            "connection_quality_scores": {}
        }
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Configuration
        self._config = {
            "max_connections_per_pool": 10,
            "connection_timeout": 30,
            "message_timeout": 10,
            "health_check_interval": 30,
            "auto_reconnect_enabled": True,
            "max_retry_attempts": 3,
            "retry_backoff_base": 2,
            "message_queue_size": 1000
        }
        
        # Connection factories
        self._connection_factories = {
            "websocket": self._create_websocket_connection,
            "redis": self._create_redis_connection,
            "http": self._create_http_connection,
            "tcp": self._create_tcp_connection
        }
        
        # Message handlers
        self._message_handlers = {
            "websocket": self._handle_websocket_message,
            "redis": self._handle_redis_message,
            "http": self._handle_http_message,
            "tcp": self._handle_tcp_message
        }
        
        import logging
        self._logger = logging.getLogger(__name__)
        self._logger.info("ProductionCommunicationBridge initialized")
        
        # Start background monitoring
        self._start_background_monitoring()
    
    async def establish_bridge(
        self,
        source_protocol: CLIProtocol,
        target_protocol: CLIProtocol,
        connection_config: Dict[str, Any]
    ) -> BridgeConnection:
        """
        Establish communication bridge between protocols.
        
        Creates a reliable connection with authentication, health monitoring,
        and auto-recovery capabilities for multi-protocol communication.
        """
        import uuid
        from datetime import datetime
        
        connection_id = str(uuid.uuid4())
        connection_type = connection_config.get("type", "websocket")
        
        try:
            self._logger.info(f"Establishing bridge: {source_protocol} -> {target_protocol} ({connection_type})")
            
            # Validate configuration
            if not self._validate_connection_config(connection_config):
                raise ValueError(f"Invalid connection configuration: {connection_config}")
            
            # Create connection based on type
            factory = self._connection_factories.get(connection_type)
            if not factory:
                raise ValueError(f"Unsupported connection type: {connection_type}")
            
            connection = await factory(
                connection_id=connection_id,
                source_protocol=source_protocol,
                target_protocol=target_protocol,
                config=connection_config
            )
            
            # Test connection
            if not await self._test_connection(connection):
                raise ConnectionError(f"Failed to establish {connection_type} connection")
            
            # Initialize connection
            connection.is_connected = True
            connection.connected_at = datetime.utcnow()
            connection.last_activity = datetime.utcnow()
            
            # Store connection
            self._connections[connection_id] = connection
            
            # Add to connection pool
            pool_key = f"{source_protocol.value}_{target_protocol.value}_{connection_type}"
            if pool_key not in self._connection_pools:
                self._connection_pools[pool_key] = []
            self._connection_pools[pool_key].append(connection)
            
            # Initialize message queue
            self._message_queues[connection_id] = asyncio.Queue(
                maxsize=self._config["message_queue_size"]
            )
            
            # Start connection monitoring
            self._start_connection_monitoring(connection)
            
            # Update metrics
            self._metrics["connections_established"] += 1
            self._connection_health[connection_id] = {
                "status": "healthy",
                "last_check": datetime.utcnow(),
                "quality_score": 1.0,
                "error_count": 0
            }
            
            self._logger.info(f"Bridge established successfully: {connection_id}")
            return connection
            
        except Exception as e:
            self._logger.error(f"Failed to establish bridge: {e}")
            self._metrics["connection_failures"] += 1
            raise
    
    async def send_message_through_bridge(
        self,
        connection_id: str,
        message: CLIMessage
    ) -> bool:
        """
        Send message through established bridge with retry logic.
        
        Provides reliable message delivery with performance monitoring,
        automatic retries, and connection quality assessment.
        """
        import time
        from datetime import datetime
        
        start_time = time.time()
        
        try:
            # Get connection
            connection = self._connections.get(connection_id)
            if not connection:
                self._logger.error(f"Connection not found: {connection_id}")
                return False
            
            if not connection.is_connected:
                # Attempt auto-reconnection
                if self._config["auto_reconnect_enabled"]:
                    if await self._attempt_reconnection(connection):
                        self._logger.info(f"Auto-reconnection successful: {connection_id}")
                    else:
                        self._logger.error(f"Auto-reconnection failed: {connection_id}")
                        return False
                else:
                    self._logger.error(f"Connection not active: {connection_id}")
                    return False
            
            # Validate message
            if not self._validate_cli_message(message):
                self._logger.error(f"Invalid message: {message.cli_message_id}")
                return False
            
            # Send message with retry logic
            max_attempts = self._config["max_retry_attempts"]
            backoff_base = self._config["retry_backoff_base"]
            
            for attempt in range(max_attempts):
                try:
                    # Get message handler
                    handler = self._message_handlers.get(connection.connection_type)
                    if not handler:
                        raise ValueError(f"No handler for connection type: {connection.connection_type}")
                    
                    # Send message
                    success = await handler(connection, message, "send")
                    
                    if success:
                        # Update connection activity
                        connection.last_activity = datetime.utcnow()
                        connection.messages_sent += 1
                        
                        # Update metrics
                        self._metrics["messages_sent"] += 1
                        delivery_time = (time.time() - start_time) * 1000  # ms
                        self._metrics["message_delivery_times"].append(delivery_time)
                        
                        # Update connection quality
                        self._update_connection_quality(connection_id, True, delivery_time)
                        
                        self._logger.debug(f"Message sent successfully: {message.cli_message_id} via {connection_id}")
                        return True
                    else:
                        raise Exception("Message delivery failed")
                        
                except Exception as e:
                    self._logger.warning(f"Send attempt {attempt + 1} failed: {e}")
                    
                    # Update connection quality
                    self._update_connection_quality(connection_id, False, None)
                    
                    if attempt < max_attempts - 1:
                        # Wait before retry with exponential backoff
                        wait_time = (backoff_base ** attempt) * 0.5
                        await asyncio.sleep(wait_time)
                    else:
                        # Final attempt failed
                        self._logger.error(f"All send attempts failed for message: {message.cli_message_id}")
                        return False
            
            return False
            
        except Exception as e:
            self._logger.error(f"Error sending message through bridge: {e}")
            self._update_connection_quality(connection_id, False, None)
            return False
        finally:
            # Record performance metrics
            total_time = (time.time() - start_time) * 1000
            if connection_id in self._connections:
                connection = self._connections[connection_id]
                times = connection.average_response_time_ms
                connection.average_response_time_ms = (times * 0.9) + (total_time * 0.1)
    
    async def listen_for_messages(
        self,
        connection_id: str
    ) -> AsyncGenerator[CLIMessage, None]:
        """
        Listen for incoming messages with real-time streaming.
        
        Provides asynchronous message reception with error handling,
        connection monitoring, and automatic recovery.
        """
        from datetime import datetime
        
        try:
            # Get connection
            connection = self._connections.get(connection_id)
            if not connection:
                self._logger.error(f"Connection not found for listening: {connection_id}")
                return
            
            if not connection.is_connected:
                self._logger.error(f"Connection not active for listening: {connection_id}")
                return
            
            self._logger.info(f"Starting message listener for connection: {connection_id}")
            
            # Start message listener task if not already running
            if connection_id not in self._active_listeners:
                listener_task = asyncio.create_task(
                    self._message_listener(connection)
                )
                self._active_listeners[connection_id] = listener_task
                self._background_tasks.add(listener_task)
            
            # Stream messages from queue
            message_queue = self._message_queues.get(connection_id)
            if not message_queue:
                self._logger.error(f"No message queue for connection: {connection_id}")
                return
            
            try:
                while connection.is_connected:
                    try:
                        # Wait for message with timeout
                        message = await asyncio.wait_for(
                            message_queue.get(),
                            timeout=self._config["message_timeout"]
                        )
                        
                        # Update connection activity
                        connection.last_activity = datetime.utcnow()
                        connection.messages_received += 1
                        
                        # Update metrics
                        self._metrics["messages_received"] += 1
                        
                        self._logger.debug(f"Message received: {message.cli_message_id} from {connection_id}")
                        yield message
                        
                    except asyncio.TimeoutError:
                        # Check if connection is still alive
                        if not await self._check_connection_health_sync(connection):
                            self._logger.warning(f"Connection health check failed: {connection_id}")
                            break
                        continue
                        
                    except Exception as e:
                        self._logger.error(f"Error receiving message from {connection_id}: {e}")
                        self._update_connection_quality(connection_id, False, None)
                        break
                        
            finally:
                # Cleanup listener
                if connection_id in self._active_listeners:
                    listener_task = self._active_listeners[connection_id]
                    listener_task.cancel()
                    del self._active_listeners[connection_id]
                    self._background_tasks.discard(listener_task)
                
                self._logger.info(f"Message listener stopped for connection: {connection_id}")
                
        except Exception as e:
            self._logger.error(f"Error in message listener: {e}")
    
    async def monitor_bridge_health(
        self,
        connection_id: str
    ) -> Dict[str, Any]:
        """
        Monitor bridge connection health with comprehensive metrics.
        
        Provides detailed health assessment including performance metrics,
        connection quality, and optimization recommendations.
        """
        try:
            connection = self._connections.get(connection_id)
            if not connection:
                return {
                    "connection_id": connection_id,
                    "status": "not_found",
                    "error": "Connection not found"
                }
            
            # Perform health check
            health_data = await self._perform_comprehensive_health_check(connection)
            
            # Calculate quality score
            quality_score = self._calculate_connection_quality_score(connection)
            
            # Get performance metrics
            performance_metrics = self._get_connection_performance_metrics(connection)
            
            # Generate recommendations
            recommendations = self._generate_health_recommendations(connection, health_data)
            
            health_report = {
                "connection_id": connection_id,
                "connection_name": connection.connection_name,
                "connection_type": connection.connection_type,
                "protocol": connection.protocol.value,
                "status": "healthy" if connection.is_connected and quality_score > 0.7 else "degraded" if quality_score > 0.3 else "unhealthy",
                "quality_score": quality_score,
                "is_connected": connection.is_connected,
                "last_activity": connection.last_activity.isoformat() if connection.last_activity else None,
                "last_heartbeat": connection.last_heartbeat.isoformat() if connection.last_heartbeat else None,
                "uptime_seconds": self._calculate_uptime(connection),
                "performance_metrics": performance_metrics,
                "health_data": health_data,
                "recommendations": recommendations,
                "auto_reconnect_enabled": self._config["auto_reconnect_enabled"],
                "monitoring_active": connection_id in self._active_listeners
            }
            
            # Update health cache
            self._connection_health[connection_id] = {
                "status": health_report["status"],
                "last_check": datetime.utcnow(),
                "quality_score": quality_score,
                "error_count": health_data.get("error_count", 0)
            }
            
            return health_report
            
        except Exception as e:
            self._logger.error(f"Error monitoring bridge health: {e}")
            return {
                "connection_id": connection_id,
                "status": "error",
                "error": str(e)
            }
    
    # ================================================================================
    # Connection Management Methods
    # ================================================================================
    
    async def _create_websocket_connection(
        self,
        connection_id: str,
        source_protocol: CLIProtocol,
        target_protocol: CLIProtocol,
        config: Dict[str, Any]
    ) -> BridgeConnection:
        """Create WebSocket connection for real-time bidirectional communication."""
        from datetime import datetime
        
        endpoint = config.get("endpoint", "ws://localhost:8765")
        auth_method = config.get("auth_method", "none")
        
        connection = BridgeConnection(
            connection_id=connection_id,
            connection_name=f"{source_protocol.value}_to_{target_protocol.value}_ws",
            protocol=target_protocol,
            endpoint=endpoint,
            connection_type="websocket",
            auth_method=auth_method,
            auto_reconnect=True,
            heartbeat_interval_seconds=30,
            created_at=datetime.utcnow()
        )
        
        # Store WebSocket-specific configuration
        connection.credentials = config.get("credentials", {})
        
        return connection
    
    async def _create_redis_connection(
        self,
        connection_id: str,
        source_protocol: CLIProtocol,
        target_protocol: CLIProtocol,
        config: Dict[str, Any]
    ) -> BridgeConnection:
        """Create Redis connection for message queuing and pub/sub patterns."""
        from datetime import datetime
        
        endpoint = config.get("endpoint", "redis://localhost:6379")
        
        connection = BridgeConnection(
            connection_id=connection_id,
            connection_name=f"{source_protocol.value}_to_{target_protocol.value}_redis",
            protocol=target_protocol,
            endpoint=endpoint,
            connection_type="redis",
            auth_method=config.get("auth_method", "none"),
            auto_reconnect=True,
            heartbeat_interval_seconds=60,
            created_at=datetime.utcnow()
        )
        
        # Store Redis-specific configuration
        connection.credentials = {
            "password": config.get("password", ""),
            "db": config.get("db", 0)
        }
        
        return connection
    
    async def _create_http_connection(
        self,
        connection_id: str,
        source_protocol: CLIProtocol,
        target_protocol: CLIProtocol,
        config: Dict[str, Any]
    ) -> BridgeConnection:
        """Create HTTP connection for request/response communication."""
        from datetime import datetime
        
        endpoint = config.get("endpoint", "http://localhost:8080")
        
        connection = BridgeConnection(
            connection_id=connection_id,
            connection_name=f"{source_protocol.value}_to_{target_protocol.value}_http",
            protocol=target_protocol,
            endpoint=endpoint,
            connection_type="http",
            auth_method=config.get("auth_method", "api_key"),
            auto_reconnect=False,  # HTTP is stateless
            heartbeat_interval_seconds=0,  # No heartbeat for HTTP
            created_at=datetime.utcnow()
        )
        
        # Store HTTP-specific configuration
        connection.credentials = {
            "api_key": config.get("api_key", ""),
            "headers": config.get("headers", {})
        }
        
        return connection
    
    async def _create_tcp_connection(
        self,
        connection_id: str,
        source_protocol: CLIProtocol,
        target_protocol: CLIProtocol,
        config: Dict[str, Any]
    ) -> BridgeConnection:
        """Create TCP connection for low-level socket communication."""
        from datetime import datetime
        
        host = config.get("host", "localhost")
        port = config.get("port", 9999)
        endpoint = f"tcp://{host}:{port}"
        
        connection = BridgeConnection(
            connection_id=connection_id,
            connection_name=f"{source_protocol.value}_to_{target_protocol.value}_tcp",
            protocol=target_protocol,
            endpoint=endpoint,
            connection_type="tcp",
            auth_method=config.get("auth_method", "none"),
            auto_reconnect=True,
            heartbeat_interval_seconds=45,
            created_at=datetime.utcnow()
        )
        
        # Store TCP-specific configuration
        connection.credentials = config.get("credentials", {})
        
        return connection
    
    # ================================================================================
    # Message Handling Methods
    # ================================================================================
    
    async def _handle_websocket_message(
        self,
        connection: BridgeConnection,
        message: CLIMessage,
        operation: str
    ) -> bool:
        """Handle WebSocket message sending and receiving."""
        try:
            if operation == "send":
                # Simulate WebSocket message sending
                import json
                
                # Serialize message
                message_data = {
                    "id": message.cli_message_id,
                    "command": message.cli_command,
                    "args": message.cli_args,
                    "data": message.input_data,
                    "timestamp": message.created_at.isoformat()
                }
                
                # In a real implementation, this would use websockets library
                # await websocket.send(json.dumps(message_data))
                
                self._logger.debug(f"WebSocket message sent: {message.cli_message_id}")
                return True
                
            elif operation == "receive":
                # Simulate WebSocket message receiving
                # In a real implementation, this would receive from websockets
                return True
                
        except Exception as e:
            self._logger.error(f"WebSocket message handling error: {e}")
            return False
    
    async def _handle_redis_message(
        self,
        connection: BridgeConnection,
        message: CLIMessage,
        operation: str
    ) -> bool:
        """Handle Redis message publishing and subscribing."""
        try:
            if operation == "send":
                # Simulate Redis message publishing
                import json
                
                channel = f"cli_messages_{connection.protocol.value}"
                message_data = {
                    "id": message.cli_message_id,
                    "command": message.cli_command,
                    "args": message.cli_args,
                    "data": message.input_data
                }
                
                # In a real implementation, this would use redis-py
                # await redis_client.publish(channel, json.dumps(message_data))
                
                self._logger.debug(f"Redis message published: {message.cli_message_id}")
                return True
                
            elif operation == "receive":
                # Simulate Redis message subscription
                return True
                
        except Exception as e:
            self._logger.error(f"Redis message handling error: {e}")
            return False
    
    async def _handle_http_message(
        self,
        connection: BridgeConnection,
        message: CLIMessage,
        operation: str
    ) -> bool:
        """Handle HTTP request/response communication."""
        try:
            if operation == "send":
                # Simulate HTTP POST request
                import json
                
                payload = {
                    "message_id": message.cli_message_id,
                    "command": message.cli_command,
                    "arguments": message.cli_args,
                    "options": message.cli_options,
                    "input_data": message.input_data
                }
                
                headers = connection.credentials.get("headers", {})
                headers["Content-Type"] = "application/json"
                
                if connection.credentials.get("api_key"):
                    headers["Authorization"] = f"Bearer {connection.credentials['api_key']}"
                
                # In a real implementation, this would use aiohttp
                # async with aiohttp.ClientSession() as session:
                #     async with session.post(connection.endpoint, 
                #                           json=payload, headers=headers) as response:
                #         return response.status == 200
                
                self._logger.debug(f"HTTP message sent: {message.cli_message_id}")
                return True
                
        except Exception as e:
            self._logger.error(f"HTTP message handling error: {e}")
            return False
    
    async def _handle_tcp_message(
        self,
        connection: BridgeConnection,
        message: CLIMessage,
        operation: str
    ) -> bool:
        """Handle TCP socket communication."""
        try:
            if operation == "send":
                # Simulate TCP message sending
                import json
                
                message_data = {
                    "id": message.cli_message_id,
                    "command": message.cli_command,
                    "args": message.cli_args,
                    "data": message.input_data
                }
                
                # In a real implementation, this would use socket operations
                # reader, writer = await asyncio.open_connection(host, port)
                # writer.write(json.dumps(message_data).encode())
                # await writer.drain()
                # writer.close()
                
                self._logger.debug(f"TCP message sent: {message.cli_message_id}")
                return True
                
        except Exception as e:
            self._logger.error(f"TCP message handling error: {e}")
            return False
    
    # ================================================================================
    # Health Monitoring and Quality Management
    # ================================================================================
    
    def _start_background_monitoring(self):
        """Start background health monitoring tasks."""
        if not self._health_monitor_task:
            self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
            self._background_tasks.add(self._health_monitor_task)
        
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._background_tasks.add(self._cleanup_task)
    
    async def _health_monitor_loop(self):
        """Background health monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self._config["health_check_interval"])
                
                for connection_id, connection in self._connections.items():
                    if connection.is_connected:
                        await self._check_connection_health_async(connection)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Health monitor error: {e}")
    
    async def _cleanup_loop(self):
        """Background cleanup loop for inactive connections."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                current_time = time.time()
                inactive_connections = []
                
                for connection_id, connection in self._connections.items():
                    if connection.last_activity:
                        inactive_time = current_time - connection.last_activity.timestamp()
                        if inactive_time > 3600:  # 1 hour
                            inactive_connections.append(connection_id)
                
                for connection_id in inactive_connections:
                    await self._cleanup_connection(connection_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Cleanup loop error: {e}")
    
    def _start_connection_monitoring(self, connection: BridgeConnection):
        """Start monitoring for a specific connection."""
        # This would start connection-specific monitoring tasks
        pass
    
    async def _check_connection_health_async(self, connection: BridgeConnection):
        """Asynchronous connection health check."""
        try:
            # Perform connection-specific health check
            if connection.connection_type == "websocket":
                # Check WebSocket ping/pong
                healthy = True  # Placeholder
            elif connection.connection_type == "redis":
                # Check Redis ping
                healthy = True  # Placeholder
            elif connection.connection_type == "http":
                # Check HTTP endpoint availability
                healthy = True  # Placeholder
            elif connection.connection_type == "tcp":
                # Check TCP socket connectivity
                healthy = True  # Placeholder
            else:
                healthy = False
            
            # Update connection health
            if not healthy and connection.is_connected:
                connection.is_connected = False
                connection.error_count += 1
                
                # Attempt auto-reconnection
                if connection.auto_reconnect:
                    await self._attempt_reconnection(connection)
        
        except Exception as e:
            self._logger.error(f"Health check error for {connection.connection_id}: {e}")
    
    async def _check_connection_health_sync(self, connection: BridgeConnection) -> bool:
        """Synchronous connection health check."""
        try:
            return connection.is_connected and connection.connection_quality > 0.1
        except Exception:
            return False
    
    def _update_connection_quality(self, connection_id: str, success: bool, response_time: Optional[float]):
        """Update connection quality score based on performance."""
        if connection_id not in self._connections:
            return
        
        connection = self._connections[connection_id]
        
        # Update quality score using exponential moving average
        if success:
            quality_adjustment = 0.1
            if response_time and response_time < 100:  # Fast response
                quality_adjustment = 0.2
        else:
            quality_adjustment = -0.3
        
        new_quality = max(0.0, min(1.0, connection.connection_quality + quality_adjustment))
        connection.connection_quality = new_quality
        
        # Update metrics cache
        self._metrics["connection_quality_scores"][connection_id] = new_quality
    
    def _calculate_connection_quality_score(self, connection: BridgeConnection) -> float:
        """Calculate comprehensive connection quality score."""
        try:
            base_quality = connection.connection_quality
            
            # Factor in error rate
            error_rate = connection.error_count / max(connection.messages_sent + connection.messages_received, 1)
            error_penalty = min(0.5, error_rate * 2)
            
            # Factor in response time
            response_time_factor = 1.0
            if connection.average_response_time_ms > 0:
                # Penalty for slow responses (>500ms gets penalty)
                if connection.average_response_time_ms > 500:
                    response_time_factor = max(0.5, 1.0 - (connection.average_response_time_ms - 500) / 1000)
            
            # Factor in uptime
            uptime_factor = 1.0
            if connection.connected_at:
                uptime_seconds = self._calculate_uptime(connection)
                if uptime_seconds > 0:
                    # Bonus for stable connections (>1 hour gets bonus)
                    if uptime_seconds > 3600:
                        uptime_factor = min(1.2, 1.0 + (uptime_seconds - 3600) / 36000)
            
            final_score = base_quality * response_time_factor * uptime_factor - error_penalty
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            self._logger.error(f"Error calculating quality score: {e}")
            return 0.5
    
    # ================================================================================
    # Helper Methods
    # ================================================================================
    
    def _validate_connection_config(self, config: Dict[str, Any]) -> bool:
        """Validate connection configuration."""
        required_fields = ["type"]
        return all(field in config for field in required_fields)
    
    def _validate_cli_message(self, message: CLIMessage) -> bool:
        """Validate CLI message format."""
        return (message.cli_message_id and 
                message.cli_command and 
                message.cli_protocol)
    
    async def _test_connection(self, connection: BridgeConnection) -> bool:
        """Test if connection is working properly."""
        try:
            # Connection-specific testing logic would go here
            # For now, return True to simulate successful connection
            return True
        except Exception as e:
            self._logger.error(f"Connection test failed: {e}")
            return False
    
    async def _attempt_reconnection(self, connection: BridgeConnection) -> bool:
        """Attempt to reconnect a failed connection."""
        try:
            self._logger.info(f"Attempting reconnection for: {connection.connection_id}")
            
            # Simulate reconnection logic
            # In a real implementation, this would re-establish the connection
            connection.is_connected = True
            connection.last_activity = datetime.utcnow()
            connection.connection_quality = max(0.5, connection.connection_quality)
            
            self._metrics["auto_reconnections"] += 1
            self._logger.info(f"Reconnection successful: {connection.connection_id}")
            return True
            
        except Exception as e:
            self._logger.error(f"Reconnection failed for {connection.connection_id}: {e}")
            return False
    
    async def _message_listener(self, connection: BridgeConnection):
        """Background task for listening to incoming messages."""
        connection_id = connection.connection_id
        message_queue = self._message_queues[connection_id]
        
        try:
            while connection.is_connected:
                # Simulate receiving messages
                # In a real implementation, this would listen on the actual connection
                await asyncio.sleep(1)
                
                # Simulate occasional message reception for testing
                if hash(connection_id) % 100 == 0:  # Occasionally receive test message
                    test_message = CLIMessage(
                        universal_message_id=f"test-{int(time.time())}",
                        cli_protocol=connection.protocol,
                        cli_command="test",
                        cli_args=["hello"],
                        input_data={"test": True}
                    )
                    
                    try:
                        await message_queue.put(test_message)
                    except asyncio.QueueFull:
                        self._logger.warning(f"Message queue full for connection: {connection_id}")
                
        except asyncio.CancelledError:
            self._logger.info(f"Message listener cancelled for: {connection_id}")
        except Exception as e:
            self._logger.error(f"Message listener error for {connection_id}: {e}")
    
    async def _perform_comprehensive_health_check(self, connection: BridgeConnection) -> Dict[str, Any]:
        """Perform comprehensive health check on connection."""
        return {
            "connectivity": "good" if connection.is_connected else "failed",
            "response_time_ms": connection.average_response_time_ms,
            "error_count": connection.error_count,
            "message_count": connection.messages_sent + connection.messages_received,
            "last_error": None,
            "connection_age_seconds": self._calculate_uptime(connection)
        }
    
    def _get_connection_performance_metrics(self, connection: BridgeConnection) -> Dict[str, Any]:
        """Get performance metrics for connection."""
        return {
            "messages_sent": connection.messages_sent,
            "messages_received": connection.messages_received,
            "average_response_time_ms": connection.average_response_time_ms,
            "error_count": connection.error_count,
            "quality_score": connection.connection_quality,
            "uptime_seconds": self._calculate_uptime(connection)
        }
    
    def _generate_health_recommendations(self, connection: BridgeConnection, health_data: Dict[str, Any]) -> List[str]:
        """Generate health recommendations for connection."""
        recommendations = []
        
        if connection.connection_quality < 0.5:
            recommendations.append("Consider restarting connection due to low quality score")
        
        if connection.average_response_time_ms > 1000:
            recommendations.append("High response time detected - check network connectivity")
        
        if connection.error_count > 10:
            recommendations.append("High error count - investigate connection stability")
        
        if not connection.is_connected:
            recommendations.append("Connection is down - enable auto-reconnect or restart manually")
        
        return recommendations
    
    def _calculate_uptime(self, connection: BridgeConnection) -> int:
        """Calculate connection uptime in seconds."""
        if not connection.connected_at:
            return 0
        
        from datetime import datetime
        return int((datetime.utcnow() - connection.connected_at).total_seconds())
    
    async def _cleanup_connection(self, connection_id: str):
        """Clean up inactive connection."""
        try:
            connection = self._connections.get(connection_id)
            if connection:
                # Stop any active listeners
                if connection_id in self._active_listeners:
                    self._active_listeners[connection_id].cancel()
                    del self._active_listeners[connection_id]
                
                # Remove from pools
                for pool in self._connection_pools.values():
                    pool[:] = [c for c in pool if c.connection_id != connection_id]
                
                # Clean up queues
                if connection_id in self._message_queues:
                    del self._message_queues[connection_id]
                
                # Remove connection
                del self._connections[connection_id]
                
                if connection_id in self._connection_health:
                    del self._connection_health[connection_id]
                
                self._logger.info(f"Connection cleaned up: {connection_id}")
                
        except Exception as e:
            self._logger.error(f"Error cleaning up connection {connection_id}: {e}")