"""
WebSocket CLI Integration for LeanVibe Agent Hive 2.0

Real-time WebSocket client integration for CLI dashboard and demo commands.
Connects to the existing WebSocket API at /api/v2/ws/{client_id} to receive
live updates about agent status, task progress, and system events.

Features:
- Real-time agent status updates
- Task progress notifications  
- System event streaming
- Automatic reconnection with exponential backoff
- Graceful error handling and fallback modes
- Dashboard integration for live data
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import websockets
import structlog
from websockets.exceptions import ConnectionClosed, WebSocketException

logger = structlog.get_logger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass
class WebSocketUpdate:
    """Structured WebSocket update message."""
    update_type: str  # agent_update, task_update, system_status
    data: Dict[str, Any]
    timestamp: datetime
    source: str = "websocket"


@dataclass
class ConnectionConfig:
    """WebSocket connection configuration."""
    base_url: str = "ws://localhost:8000"
    endpoint: str = "/api/v2/ws"
    reconnect_attempts: int = 5
    reconnect_delay: float = 1.0  # Initial delay in seconds
    reconnect_backoff: float = 2.0  # Backoff multiplier
    ping_interval: float = 30.0  # Ping interval in seconds
    ping_timeout: float = 10.0  # Ping timeout in seconds
    subscribe_to_all: bool = True
    subscriptions: List[str] = field(default_factory=list)


class CLIWebSocketClient:
    """WebSocket client for CLI real-time updates."""
    
    def __init__(self, config: Optional[ConnectionConfig] = None):
        self.config = config or ConnectionConfig()
        self.client_id = str(uuid.uuid4())[:8]  # Short client ID for CLI
        self.connection_state = ConnectionState.DISCONNECTED
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.update_handlers: List[Callable[[WebSocketUpdate], None]] = []
        self.reconnect_count = 0
        self.last_update = datetime.utcnow()
        self.is_running = False
        self._connection_task: Optional[asyncio.Task] = None
        
        # Message queue for updates when dashboard is not ready
        self.update_queue: asyncio.Queue = asyncio.Queue()
        self.max_queue_size = 100
    
    async def connect(self) -> bool:
        """Connect to the WebSocket server."""
        if self.connection_state in [ConnectionState.CONNECTED, ConnectionState.CONNECTING]:
            return True
        
        self.connection_state = ConnectionState.CONNECTING
        ws_url = f"{self.config.base_url}{self.config.endpoint}/{self.client_id}"
        
        try:
            logger.info(f"Connecting to WebSocket: {ws_url}")
            
            self.websocket = await websockets.connect(
                ws_url,
                ping_interval=self.config.ping_interval,
                ping_timeout=self.config.ping_timeout,
                close_timeout=5.0
            )
            
            self.connection_state = ConnectionState.CONNECTED
            self.reconnect_count = 0
            
            logger.info(f"WebSocket connected successfully", client_id=self.client_id)
            
            # Setup subscriptions
            await self._setup_subscriptions()
            
            return True
            
        except Exception as e:
            self.connection_state = ConnectionState.FAILED
            logger.error(f"WebSocket connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket server."""
        self.is_running = False
        
        if self._connection_task:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
        
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
            finally:
                self.websocket = None
        
        self.connection_state = ConnectionState.DISCONNECTED
        logger.info("WebSocket disconnected")
    
    async def _setup_subscriptions(self):
        """Setup WebSocket subscriptions."""
        if not self.websocket:
            return
        
        try:
            # Subscribe to all updates if configured
            if self.config.subscribe_to_all:
                await self._send_command({
                    "command": "subscribe_agent",
                    "agent_id": "*"
                })
                
                await self._send_command({
                    "command": "subscribe_task", 
                    "task_id": "*"
                })
            
            # Setup specific subscriptions
            for subscription in self.config.subscriptions:
                if subscription.startswith("agent:"):
                    agent_id = subscription.replace("agent:", "")
                    await self._send_command({
                        "command": "subscribe_agent",
                        "agent_id": agent_id
                    })
                elif subscription.startswith("task:"):
                    task_id = subscription.replace("task:", "")
                    await self._send_command({
                        "command": "subscribe_task",
                        "task_id": task_id
                    })
            
            logger.info(f"WebSocket subscriptions setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup subscriptions: {e}")
    
    async def _send_command(self, command: Dict[str, Any]):
        """Send command to WebSocket server."""
        if not self.websocket:
            return
        
        try:
            await self.websocket.send(json.dumps(command))
        except Exception as e:
            logger.error(f"Failed to send WebSocket command: {e}")
    
    async def start_listening(self):
        """Start listening for WebSocket messages."""
        self.is_running = True
        self._connection_task = asyncio.create_task(self._listen_loop())
        return self._connection_task
    
    async def _listen_loop(self):
        """Main WebSocket listening loop with reconnection."""
        while self.is_running:
            try:
                if not await self.connect():
                    await self._handle_reconnection()
                    continue
                
                await self._message_loop()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WebSocket listen loop error: {e}")
                await self._handle_reconnection()
    
    async def _message_loop(self):
        """Process WebSocket messages."""
        if not self.websocket:
            return
        
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._process_message(data)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON received: {e}")
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
                    
        except ConnectionClosed:
            logger.info("WebSocket connection closed")
            self.connection_state = ConnectionState.DISCONNECTED
        except WebSocketException as e:
            logger.error(f"WebSocket error: {e}")
            self.connection_state = ConnectionState.FAILED
    
    async def _process_message(self, data: Dict[str, Any]):
        """Process incoming WebSocket message."""
        message_type = data.get("type")
        if not message_type:
            return
        
        # Create structured update
        update = WebSocketUpdate(
            update_type=message_type,
            data=data.get("data", {}),
            timestamp=datetime.utcnow()
        )
        
        self.last_update = update.timestamp
        
        # Add to queue if it's getting full, remove oldest items
        if self.update_queue.qsize() >= self.max_queue_size:
            try:
                self.update_queue.get_nowait()  # Remove oldest
            except asyncio.QueueEmpty:
                pass
        
        await self.update_queue.put(update)
        
        # Notify handlers
        for handler in self.update_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(update)
                else:
                    handler(update)
            except Exception as e:
                logger.error(f"Error in update handler: {e}")
        
        logger.debug(f"Processed WebSocket update: {message_type}")
    
    async def _handle_reconnection(self):
        """Handle WebSocket reconnection with exponential backoff."""
        if not self.is_running:
            return
        
        if self.reconnect_count >= self.config.reconnect_attempts:
            logger.error("Max reconnection attempts reached, giving up")
            self.connection_state = ConnectionState.FAILED
            return
        
        self.connection_state = ConnectionState.RECONNECTING
        self.reconnect_count += 1
        
        delay = self.config.reconnect_delay * (self.config.reconnect_backoff ** (self.reconnect_count - 1))
        delay = min(delay, 60.0)  # Cap at 60 seconds
        
        logger.info(f"Reconnecting in {delay:.1f} seconds (attempt {self.reconnect_count}/{self.config.reconnect_attempts})")
        
        await asyncio.sleep(delay)
    
    def add_update_handler(self, handler: Callable[[WebSocketUpdate], None]):
        """Add handler for WebSocket updates."""
        self.update_handlers.append(handler)
    
    def remove_update_handler(self, handler: Callable[[WebSocketUpdate], None]):
        """Remove update handler."""
        if handler in self.update_handlers:
            self.update_handlers.remove(handler)
    
    async def get_updates(self, timeout: Optional[float] = None) -> Optional[WebSocketUpdate]:
        """Get next update from queue."""
        try:
            if timeout:
                return await asyncio.wait_for(self.update_queue.get(), timeout=timeout)
            else:
                return await self.update_queue.get()
        except asyncio.TimeoutError:
            return None
        except asyncio.QueueEmpty:
            return None
    
    async def get_system_status(self) -> Optional[Dict[str, Any]]:
        """Request current system status via WebSocket."""
        if not self.websocket:
            return None
        
        try:
            await self._send_command({"command": "get_system_status"})
            
            # Wait for system_status response
            timeout = 5.0
            start_time = asyncio.get_event_loop().time()
            
            while (asyncio.get_event_loop().time() - start_time) < timeout:
                try:
                    update = await asyncio.wait_for(self.update_queue.get(), timeout=0.5)
                    if update.update_type == "system_status":
                        return update.data
                    else:
                        # Put back non-matching update
                        await self.update_queue.put(update)
                except asyncio.TimeoutError:
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return None
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection status information."""
        return {
            "client_id": self.client_id,
            "state": self.connection_state.value,
            "reconnect_count": self.reconnect_count,
            "last_update": self.last_update.isoformat(),
            "queue_size": self.update_queue.qsize(),
            "handlers_count": len(self.update_handlers)
        }


class WebSocketDashboardIntegration:
    """Integration between WebSocket client and CLI dashboard."""
    
    def __init__(self, dashboard, websocket_client: Optional[CLIWebSocketClient] = None):
        self.dashboard = dashboard
        self.websocket_client = websocket_client or CLIWebSocketClient()
        self.last_agent_data = {}
        self.last_task_data = {}
        self.last_system_data = {}
        
        # Setup update handlers
        self.websocket_client.add_update_handler(self._handle_websocket_update)
    
    async def start_integration(self):
        """Start WebSocket integration with dashboard."""
        try:
            # Start WebSocket listening
            await self.websocket_client.start_listening()
            logger.info("WebSocket dashboard integration started")
            return True
        except Exception as e:
            logger.error(f"Failed to start WebSocket integration: {e}")
            return False
    
    async def stop_integration(self):
        """Stop WebSocket integration."""
        await self.websocket_client.disconnect()
        logger.info("WebSocket dashboard integration stopped")
    
    async def _handle_websocket_update(self, update: WebSocketUpdate):
        """Handle WebSocket updates for dashboard."""
        try:
            if update.update_type == "agent_update":
                self.last_agent_data.update(update.data)
                # Could trigger dashboard refresh or update specific components
                
            elif update.update_type == "task_update":
                self.last_task_data.update(update.data)
                # Update task progress in dashboard
                
            elif update.update_type == "system_status":
                self.last_system_data.update(update.data)
                # Update system metrics
                
            # Log for debugging
            logger.debug(f"Dashboard received {update.update_type} update")
            
        except Exception as e:
            logger.error(f"Error handling WebSocket update in dashboard: {e}")
    
    def get_latest_data(self) -> Dict[str, Any]:
        """Get latest data from WebSocket updates."""
        return {
            "agents": self.last_agent_data,
            "tasks": self.last_task_data,
            "system": self.last_system_data,
            "connection": self.websocket_client.get_connection_info()
        }


# Factory functions for easy client creation
def create_websocket_client(
    base_url: str = "ws://localhost:8000",
    subscribe_to_all: bool = True
) -> CLIWebSocketClient:
    """Create WebSocket client with configuration."""
    config = ConnectionConfig(
        base_url=base_url,
        subscribe_to_all=subscribe_to_all
    )
    return CLIWebSocketClient(config)


def create_dashboard_integration(
    dashboard, 
    websocket_client: Optional[CLIWebSocketClient] = None
) -> WebSocketDashboardIntegration:
    """Create WebSocket dashboard integration."""
    return WebSocketDashboardIntegration(dashboard, websocket_client)


# Example usage functions
async def test_websocket_connection(base_url: str = "ws://localhost:8000"):
    """Test WebSocket connection and print updates."""
    client = create_websocket_client(base_url)
    
    def print_update(update: WebSocketUpdate):
        print(f"[{update.timestamp.strftime('%H:%M:%S')}] {update.update_type}: {update.data}")
    
    client.add_update_handler(print_update)
    
    try:
        await client.start_listening()
        
        # Keep running for testing
        while client.is_running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("Stopping WebSocket test...")
    finally:
        await client.disconnect()


# Export main classes
__all__ = [
    'CLIWebSocketClient',
    'WebSocketDashboardIntegration', 
    'WebSocketUpdate',
    'ConnectionConfig',
    'ConnectionState',
    'create_websocket_client',
    'create_dashboard_integration',
    'test_websocket_connection'
]