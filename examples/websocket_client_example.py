import asyncio
"""
Example WebSocket Client for Project Index Events

This example demonstrates how to connect to and interact with the Project Index
WebSocket system, including subscription management, event handling, and error recovery.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass

import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    """Client configuration settings."""
    websocket_url: str = "ws://localhost:8000/api/v1/project-index/websocket"
    user_id: str = "example-user"
    project_id: Optional[str] = None
    auto_reconnect: bool = True
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 10
    heartbeat_interval: float = 30.0
    event_handlers: Dict[str, Callable] = None
    
    def __post_init__(self):
        if self.event_handlers is None:
            self.event_handlers = {}


class ProjectIndexWebSocketClient:
    """
    WebSocket client for Project Index events.
    
    Provides high-level interface for connecting to the WebSocket system,
    managing subscriptions, and handling events with automatic reconnection.
    """
    
    def __init__(self, config: ClientConfig):
        self.config = config
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.connected = False
        self.reconnect_attempts = 0
        
        # Event handling
        self.subscribed_events: List[str] = []
        self.subscribed_projects: List[str] = []
        self.user_preferences: Dict[str, Any] = {}
        
        # Background tasks
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.reconnect_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.events_received = 0
        self.errors_encountered = 0
        self.connection_count = 0
        self.last_event_time: Optional[datetime] = None
        
        # Default event handlers
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Set up default event handlers."""
        self.config.event_handlers.update({
            "welcome": self._handle_welcome,
            "subscription_ack": self._handle_subscription_ack,
            "error": self._handle_error,
            "pong": self._handle_pong,
            "replay_complete": self._handle_replay_complete
        })
    
    async def connect(self) -> bool:
        """
        Connect to the WebSocket server.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"Connecting to WebSocket: {self.config.websocket_url}")
            
            # Create WebSocket connection
            self.websocket = await websockets.connect(
                self.config.websocket_url,
                extra_headers={"User-Id": self.config.user_id}
            )
            
            self.connected = True
            self.reconnect_attempts = 0
            self.connection_count += 1
            
            logger.info("WebSocket connected successfully")
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Start message handling loop
            asyncio.create_task(self._message_loop())
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self.connected = False
            
            # Attempt reconnection if enabled
            if self.config.auto_reconnect:
                await self._schedule_reconnect()
            
            return False
    
    async def disconnect(self):
        """Disconnect from the WebSocket server."""
        self.connected = False
        
        # Cancel background tasks
        await self._stop_background_tasks()
        
        # Close WebSocket connection
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        logger.info("WebSocket disconnected")
    
    async def subscribe_to_events(
        self, 
        event_types: List[str], 
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Subscribe to specific event types.
        
        Args:
            event_types: List of event types to subscribe to
            project_id: Optional project ID to filter events
            session_id: Optional session ID to filter events
            filters: Optional additional filters
            
        Returns:
            True if subscription successful, False otherwise
        """
        if not self.connected:
            logger.error("Cannot subscribe: not connected")
            return False
        
        try:
            subscription_message = {
                "action": "subscribe",
                "event_types": event_types,
                "project_id": project_id or self.config.project_id,
                "session_id": session_id,
                "filters": filters or {}
            }
            
            await self._send_message(subscription_message)
            
            # Update local subscription state
            self.subscribed_events.extend(event_types)
            if project_id:
                if project_id not in self.subscribed_projects:
                    self.subscribed_projects.append(project_id)
            
            logger.info(f"Subscribed to events: {event_types}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to events: {e}")
            return False
    
    async def unsubscribe_from_events(
        self, 
        event_types: Optional[List[str]] = None
    ) -> bool:
        """
        Unsubscribe from events.
        
        Args:
            event_types: Specific event types to unsubscribe from, or None for all
            
        Returns:
            True if unsubscription successful, False otherwise
        """
        if not self.connected:
            logger.error("Cannot unsubscribe: not connected")
            return False
        
        try:
            unsubscribe_message = {
                "action": "unsubscribe",
                "event_types": event_types or self.subscribed_events
            }
            
            await self._send_message(unsubscribe_message)
            
            # Update local subscription state
            if event_types:
                for event_type in event_types:
                    if event_type in self.subscribed_events:
                        self.subscribed_events.remove(event_type)
            else:
                self.subscribed_events.clear()
                self.subscribed_projects.clear()
            
            logger.info(f"Unsubscribed from events: {event_types or 'all'}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe from events: {e}")
            return False
    
    async def set_user_preferences(self, preferences: Dict[str, Any]) -> bool:
        """
        Set user preferences for event filtering.
        
        Args:
            preferences: User preference dictionary
            
        Returns:
            True if preferences set successfully, False otherwise
        """
        if not self.connected:
            logger.error("Cannot set preferences: not connected")
            return False
        
        try:
            preferences_message = {
                "action": "set_preferences",
                "preferences": preferences
            }
            
            await self._send_message(preferences_message)
            self.user_preferences.update(preferences)
            
            logger.info("User preferences updated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set user preferences: {e}")
            return False
    
    async def request_replay(
        self,
        project_id: Optional[str] = None,
        since: Optional[datetime] = None,
        event_types: Optional[List[str]] = None,
        max_events: int = 50
    ) -> bool:
        """
        Request event replay for missed events.
        
        Args:
            project_id: Project to replay events for
            since: Timestamp to replay events since
            event_types: Specific event types to replay
            max_events: Maximum number of events to replay
            
        Returns:
            True if replay request successful, False otherwise
        """
        if not self.connected:
            logger.error("Cannot request replay: not connected")
            return False
        
        try:
            replay_message = {
                "action": "replay",
                "project_id": project_id or self.config.project_id,
                "since": since.isoformat() if since else None,
                "event_types": event_types,
                "max_events": max_events,
                "include_delivered": False
            }
            
            await self._send_message(replay_message)
            
            logger.info(f"Requested event replay for project: {project_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to request replay: {e}")
            return False
    
    async def ping(self) -> bool:
        """
        Send ping to test connection.
        
        Returns:
            True if ping successful, False otherwise
        """
        if not self.connected:
            return False
        
        try:
            ping_message = {
                "action": "ping",
                "timestamp": datetime.utcnow().isoformat(),
                "correlation_id": str(uuid.uuid4())
            }
            
            await self._send_message(ping_message)
            return True
            
        except Exception as e:
            logger.error(f"Failed to send ping: {e}")
            return False
    
    def add_event_handler(self, event_type: str, handler: Callable[[Dict[str, Any]], None]):
        """
        Add custom event handler.
        
        Args:
            event_type: Event type to handle
            handler: Handler function that takes event data
        """
        self.config.event_handlers[event_type] = handler
        logger.info(f"Added event handler for: {event_type}")
    
    def remove_event_handler(self, event_type: str):
        """
        Remove event handler.
        
        Args:
            event_type: Event type to remove handler for
        """
        if event_type in self.config.event_handlers:
            del self.config.event_handlers[event_type]
            logger.info(f"Removed event handler for: {event_type}")
    
    async def _send_message(self, message: Dict[str, Any]):
        """Send message to WebSocket server."""
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")
        
        message_json = json.dumps(message, default=str)
        await self.websocket.send(message_json)
    
    async def _message_loop(self):
        """Main message handling loop."""
        try:
            while self.connected and self.websocket:
                try:
                    # Receive message with timeout
                    message_data = await asyncio.wait_for(
                        self.websocket.recv(), 
                        timeout=60.0
                    )
                    
                    # Parse and handle message
                    await self._handle_message(message_data)
                    
                except asyncio.TimeoutError:
                    # Timeout - send ping to keep connection alive
                    await self.ping()
                    continue
                    
                except (ConnectionClosedError, ConnectionClosedOK):
                    logger.warning("WebSocket connection closed")
                    break
                    
        except Exception as e:
            logger.error(f"Message loop error: {e}")
            self.errors_encountered += 1
        
        finally:
            self.connected = False
            
            # Attempt reconnection if enabled
            if self.config.auto_reconnect:
                await self._schedule_reconnect()
    
    async def _handle_message(self, message_data: str):
        """Handle incoming WebSocket message."""
        try:
            message = json.loads(message_data)
            event_type = message.get("type")
            
            if not event_type:
                logger.warning("Received message without type field")
                return
            
            # Update metrics
            self.events_received += 1
            self.last_event_time = datetime.utcnow()
            
            # Call event handler if registered
            if event_type in self.config.event_handlers:
                try:
                    handler = self.config.event_handlers[event_type]
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message)
                    else:
                        handler(message)
                except Exception as e:
                    logger.error(f"Event handler error for {event_type}: {e}")
            else:
                logger.debug(f"No handler for event type: {event_type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message JSON: {e}")
            self.errors_encountered += 1
        except Exception as e:
            logger.error(f"Message handling error: {e}")
            self.errors_encountered += 1
    
    async def _start_background_tasks(self):
        """Start background tasks."""
        # Start heartbeat task
        if self.config.heartbeat_interval > 0:
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
    
    async def _stop_background_tasks(self):
        """Stop background tasks."""
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
            self.heartbeat_task = None
        
        if self.reconnect_task:
            self.reconnect_task.cancel()
            try:
                await self.reconnect_task
            except asyncio.CancelledError:
                pass
            self.reconnect_task = None
    
    async def _heartbeat_loop(self):
        """Background heartbeat loop."""
        try:
            while self.connected:
                await asyncio.sleep(self.config.heartbeat_interval)
                if self.connected:
                    await self.ping()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Heartbeat loop error: {e}")
    
    async def _schedule_reconnect(self):
        """Schedule reconnection attempt."""
        if self.reconnect_attempts >= self.config.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            return
        
        self.reconnect_attempts += 1
        
        logger.info(f"Scheduling reconnection attempt {self.reconnect_attempts} in {self.config.reconnect_delay}s")
        
        self.reconnect_task = asyncio.create_task(self._reconnect_after_delay())
    
    async def _reconnect_after_delay(self):
        """Reconnect after delay."""
        try:
            await asyncio.sleep(self.config.reconnect_delay)
            
            # Attempt reconnection
            if await self.connect():
                # Re-establish subscriptions
                if self.subscribed_events:
                    await self.subscribe_to_events(
                        self.subscribed_events,
                        self.subscribed_projects[0] if self.subscribed_projects else None
                    )
                
                # Re-apply user preferences
                if self.user_preferences:
                    await self.set_user_preferences(self.user_preferences)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Reconnection error: {e}")
            
            # Schedule another attempt
            if self.config.auto_reconnect:
                await self._schedule_reconnect()
    
    # Default event handlers
    
    async def _handle_welcome(self, message: Dict[str, Any]):
        """Handle welcome message."""
        data = message.get("data", {})
        logger.info(f"Received welcome message: {data.get('connection_id')}")
        
        # Log available events
        available_events = data.get("available_events", [])
        logger.info(f"Available event types: {available_events}")
    
    async def _handle_subscription_ack(self, message: Dict[str, Any]):
        """Handle subscription acknowledgment."""
        data = message.get("data", {})
        action = data.get("action")
        status = data.get("status")
        event_types = data.get("event_types", [])
        
        logger.info(f"Subscription {action} {status} for events: {event_types}")
    
    async def _handle_error(self, message: Dict[str, Any]):
        """Handle error message."""
        data = message.get("data", {})
        error_type = data.get("error", "UNKNOWN")
        error_message = data.get("message", "No message")
        error_details = data.get("details", "")
        
        logger.error(f"WebSocket error: {error_type} - {error_message} ({error_details})")
        self.errors_encountered += 1
    
    async def _handle_pong(self, message: Dict[str, Any]):
        """Handle pong response."""
        data = message.get("data", {})
        health = data.get("connection_health", 0.0)
        logger.debug(f"Pong received - connection health: {health}")
    
    async def _handle_replay_complete(self, message: Dict[str, Any]):
        """Handle replay completion."""
        data = message.get("data", {})
        events_replayed = data.get("events_replayed", 0)
        project_id = data.get("project_id")
        
        logger.info(f"Event replay completed: {events_replayed} events for project {project_id}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "connected": self.connected,
            "connection_count": self.connection_count,
            "events_received": self.events_received,
            "errors_encountered": self.errors_encountered,
            "reconnect_attempts": self.reconnect_attempts,
            "subscribed_events": self.subscribed_events,
            "subscribed_projects": self.subscribed_projects,
            "last_event_time": self.last_event_time.isoformat() if self.last_event_time else None,
            "user_preferences": self.user_preferences
        }


# Example usage and event handlers

def handle_project_updated(message: Dict[str, Any]):
    """Handle project index updated events."""
    data = message.get("data", {})
    project_name = data.get("project_name")
    files_analyzed = data.get("files_analyzed")
    status = data.get("status")
    
    print(f"📊 Project '{project_name}' updated: {files_analyzed} files analyzed, status: {status}")
    
    # Update UI or trigger actions based on project completion
    if status == "completed":
        print(f"✅ Analysis completed for {project_name}")
    elif status == "failed":
        print(f"❌ Analysis failed for {project_name}")


def handle_analysis_progress(message: Dict[str, Any]):
    """Handle analysis progress events."""
    data = message.get("data", {})
    progress = data.get("progress_percentage", 0)
    files_processed = data.get("files_processed", 0)
    total_files = data.get("total_files", 0)
    current_file = data.get("current_file", "")
    
    print(f"🔄 Analysis Progress: {progress}% ({files_processed}/{total_files}) - {current_file}")
    
    # Update progress bar or status display


def handle_dependency_changed(message: Dict[str, Any]):
    """Handle dependency change events."""
    data = message.get("data", {})
    file_path = data.get("file_path")
    change_type = data.get("change_type")
    impact = data.get("impact_analysis", {})
    affected_files = impact.get("affected_files", [])
    
    print(f"🔗 Dependency changed: {file_path} ({change_type})")
    if affected_files:
        print(f"   Affected files: {', '.join(affected_files)}")


def handle_context_optimized(message: Dict[str, Any]):
    """Handle context optimization events."""
    data = message.get("data", {})
    task_type = data.get("task_type")
    results = data.get("optimization_results", {})
    confidence = results.get("confidence_score", 0.0)
    selected_files = results.get("selected_files", 0)
    
    print(f"🧠 Context optimized for {task_type}: {selected_files} files selected (confidence: {confidence:.2f})")


async def main():
    """Example main function demonstrating client usage."""
    # Configure client
    config = ClientConfig(
        websocket_url="ws://localhost:8000/api/v1/project-index/websocket",
        user_id="example-user",
        project_id="your-project-uuid",
        auto_reconnect=True,
        reconnect_delay=5.0,
        heartbeat_interval=30.0
    )
    
    # Create client
    client = ProjectIndexWebSocketClient(config)
    
    # Add custom event handlers
    client.add_event_handler("project_index_updated", handle_project_updated)
    client.add_event_handler("analysis_progress", handle_analysis_progress)
    client.add_event_handler("dependency_changed", handle_dependency_changed)
    client.add_event_handler("context_optimized", handle_context_optimized)
    
    try:
        # Connect to WebSocket
        if await client.connect():
            print("✅ Connected to WebSocket")
            
            # Set user preferences
            await client.set_user_preferences({
                "preferred_languages": ["python", "javascript"],
                "ignored_file_patterns": ["*.test.js", "*.spec.py"],
                "min_progress_updates": 25,
                "high_impact_only": False,
                "notification_frequency": "normal"
            })
            
            # Subscribe to events
            await client.subscribe_to_events([
                "project_index_updated",
                "analysis_progress",
                "dependency_changed",
                "context_optimized"
            ])
            
            # Request replay of recent events
            await client.request_replay(since=datetime.utcnow().replace(hour=0, minute=0, second=0))
            
            print("🔄 Listening for events... Press Ctrl+C to stop")
            
            # Keep the client running
            while True:
                await asyncio.sleep(10)
                
                # Print statistics periodically
                stats = client.get_statistics()
                print(f"📈 Stats: {stats['events_received']} events received, "
                      f"{stats['errors_encountered']} errors, connected: {stats['connected']}")
        
        else:
            print("❌ Failed to connect to WebSocket")
    
    except KeyboardInterrupt:
        print("\n🛑 Shutting down client...")
    
    finally:
        await client.disconnect()
        print("👋 Client disconnected")


if __name__ == "__main__":
    asyncio.run(main())