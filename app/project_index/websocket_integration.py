"""
Enhanced WebSocket Integration for Project Index Events

This module extends the existing WebSocket infrastructure to support the new 
project index event types with advanced filtering, batching, and performance optimization.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Set, Any, Callable
from uuid import UUID

import structlog
from fastapi import WebSocket, WebSocketDisconnect, Depends, HTTPException
from pydantic import BaseModel, ValidationError

from ..core.redis import get_redis_client, RedisClient
from .websocket_events import (
    ProjectIndexEventPublisher, get_event_publisher,
    ProjectIndexEventType, ProjectIndexWebSocketEvent
)
from .event_filters import EventFilter, get_event_filter, UserPreferences, FilterRule
from .event_history import EventHistoryManager, get_event_history_manager, ReplayRequest
from .websocket_performance import WebSocketPerformanceManager, get_performance_manager, EventPriority

logger = structlog.get_logger()


class EnhancedSubscriptionRequest(BaseModel):
    """Enhanced subscription request with filtering options."""
    action: str  # "subscribe" or "unsubscribe"
    event_types: List[str]
    project_id: Optional[str] = None
    session_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    user_preferences: Optional[Dict[str, Any]] = None


class ProjectIndexWebSocketManager:
    """Enhanced WebSocket manager for project index events."""
    
    def __init__(self, redis_client: Optional[RedisClient] = None):
        self.redis_client = redis_client or get_redis_client()
        
        # Core components
        self.event_publisher = get_event_publisher()
        self.event_filter = get_event_filter()
        self.history_manager = None  # Will be initialized async
        self.performance_manager = get_performance_manager()
        
        # Connection management
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_users: Dict[str, str] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Event delivery tracking
        self.delivery_callbacks: List[Callable] = []
        
        # Redis subscriber for distributed events
        self.redis_subscriber_task: Optional[asyncio.Task] = None
        
        # Performance metrics
        self.metrics = {
            "connections_handled": 0,
            "events_delivered": 0,
            "events_filtered": 0,
            "reconnections": 0,
            "errors": 0
        }
    
    async def initialize(self):
        """Initialize async components."""
        self.history_manager = await get_event_history_manager()
        
        # Start Redis subscriber for distributed events
        if not self.redis_subscriber_task:
            self.redis_subscriber_task = asyncio.create_task(self._redis_event_listener())
    
    async def handle_connection(self, websocket: WebSocket, user_id: str) -> None:
        """Handle new WebSocket connection with enhanced features."""
        connection_id = str(uuid.uuid4())
        
        try:
            await websocket.accept()
            
            # Register connection
            self.active_connections[connection_id] = websocket
            self.connection_users[connection_id] = user_id
            self.connection_metadata[connection_id] = {
                "connected_at": datetime.utcnow(),
                "user_id": user_id,
                "subscriptions": set(),
                "last_activity": datetime.utcnow()
            }
            
            # Add to performance manager
            self.performance_manager.add_connection(connection_id)
            
            # Subscribe to WebSocket events via event publisher
            await self.event_publisher.subscribe_to_events(
                connection_id, 
                [et.value for et in ProjectIndexEventType]
            )
            
            self.metrics["connections_handled"] += 1
            
            # Send welcome message with enhanced capabilities
            welcome_message = {
                "type": "welcome",
                "data": {
                    "connection_id": connection_id,
                    "user_id": user_id,
                    "server_capabilities": {
                        "event_types": [et.value for et in ProjectIndexEventType],
                        "filtering": True,
                        "batching": True,
                        "compression": True,
                        "replay": True,
                        "rate_limiting": True
                    },
                    "performance_info": self.performance_manager.get_performance_summary()
                },
                "timestamp": datetime.utcnow().isoformat(),
                "correlation_id": str(uuid.uuid4())
            }
            
            await self._send_to_connection(connection_id, welcome_message)
            
            # Send recent events if any
            await self._send_replay_events(connection_id, user_id)
            
            logger.info("WebSocket connection established", 
                       connection_id=connection_id, user_id=user_id)
            
            # Listen for messages
            await self._message_loop(connection_id, websocket)
            
        except Exception as e:
            logger.error("WebSocket connection error", 
                        connection_id=connection_id, error=str(e))
            self.metrics["errors"] += 1
        finally:
            await self._cleanup_connection(connection_id)
    
    async def _message_loop(self, connection_id: str, websocket: WebSocket) -> None:
        """Main message handling loop for WebSocket connection."""
        try:
            while True:
                try:
                    # Update last activity
                    if connection_id in self.connection_metadata:
                        self.connection_metadata[connection_id]["last_activity"] = datetime.utcnow()
                    
                    # Receive message
                    data = await websocket.receive_text()
                    await self._handle_message(connection_id, data)
                    
                except WebSocketDisconnect:
                    logger.info("WebSocket disconnected", connection_id=connection_id)
                    break
                except Exception as e:
                    logger.error("Message handling error", 
                               connection_id=connection_id, error=str(e))
                    
                    # Send error response
                    error_message = {
                        "type": "error",
                        "data": {
                            "error": "MESSAGE_PROCESSING_ERROR",
                            "message": "Failed to process message",
                            "details": str(e)
                        },
                        "timestamp": datetime.utcnow().isoformat(),
                        "correlation_id": str(uuid.uuid4())
                    }
                    
                    await self._send_to_connection(connection_id, error_message)
                    
        except Exception as e:
            logger.error("Message loop error", connection_id=connection_id, error=str(e))
    
    async def _handle_message(self, connection_id: str, data: str) -> None:
        """Handle incoming WebSocket message."""
        try:
            message_data = json.loads(data)
            
            # Validate message structure
            if "action" not in message_data:
                raise ValueError("Missing 'action' field in message")
            
            action = message_data["action"]
            
            if action == "subscribe":
                await self._handle_subscription(connection_id, message_data)
            elif action == "unsubscribe":
                await self._handle_unsubscription(connection_id, message_data)
            elif action == "replay":
                await self._handle_replay_request(connection_id, message_data)
            elif action == "set_preferences":
                await self._handle_user_preferences(connection_id, message_data)
            elif action == "ping":
                await self._handle_ping(connection_id, message_data)
            elif action == "get_stats":
                await self._handle_stats_request(connection_id)
            else:
                raise ValueError(f"Unknown action: {action}")
            
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            error_message = {
                "type": "error",
                "data": {
                    "error": "INVALID_MESSAGE",
                    "message": "Invalid message format or content",
                    "details": str(e)
                },
                "timestamp": datetime.utcnow().isoformat(),
                "correlation_id": str(uuid.uuid4())
            }
            await self._send_to_connection(connection_id, error_message)
        
        except Exception as e:
            logger.error("Message handling failed", 
                        connection_id=connection_id, error=str(e))
    
    async def _handle_subscription(self, connection_id: str, message_data: Dict[str, Any]) -> None:
        """Handle subscription request."""
        try:
            event_types = message_data.get("event_types", [])
            project_id = message_data.get("project_id")
            session_id = message_data.get("session_id")
            filters = message_data.get("filters", {})
            
            # Subscribe to event types
            await self.event_publisher.subscribe_to_events(connection_id, event_types)
            
            # Subscribe to project if specified
            if project_id:
                await self.event_publisher.subscribe_to_project(connection_id, UUID(project_id))
            
            # Update connection metadata
            if connection_id in self.connection_metadata:
                self.connection_metadata[connection_id]["subscriptions"].update(event_types)
                if project_id:
                    self.connection_metadata[connection_id]["project_id"] = project_id
                if session_id:
                    self.connection_metadata[connection_id]["session_id"] = session_id
                if filters:
                    self.connection_metadata[connection_id]["filters"] = filters
            
            # Send acknowledgment
            ack_message = {
                "type": "subscription_ack",
                "data": {
                    "action": "subscribe",
                    "event_types": event_types,
                    "project_id": project_id,
                    "session_id": session_id,
                    "status": "success"
                },
                "timestamp": datetime.utcnow().isoformat(),
                "correlation_id": str(uuid.uuid4())
            }
            
            await self._send_to_connection(connection_id, ack_message)
            
            logger.info("Subscription processed", 
                       connection_id=connection_id,
                       event_types=event_types,
                       project_id=project_id)
            
        except Exception as e:
            logger.error("Subscription handling failed", 
                        connection_id=connection_id, error=str(e))
    
    async def _handle_unsubscription(self, connection_id: str, message_data: Dict[str, Any]) -> None:
        """Handle unsubscription request."""
        try:
            # Unsubscribe from all events for this connection
            await self.event_publisher.unsubscribe_client(connection_id)
            
            # Update connection metadata
            if connection_id in self.connection_metadata:
                self.connection_metadata[connection_id]["subscriptions"].clear()
            
            # Send acknowledgment
            ack_message = {
                "type": "subscription_ack",
                "data": {
                    "action": "unsubscribe",
                    "status": "success"
                },
                "timestamp": datetime.utcnow().isoformat(),
                "correlation_id": str(uuid.uuid4())
            }
            
            await self._send_to_connection(connection_id, ack_message)
            
        except Exception as e:
            logger.error("Unsubscription handling failed", 
                        connection_id=connection_id, error=str(e))
    
    async def _handle_replay_request(self, connection_id: str, message_data: Dict[str, Any]) -> None:
        """Handle event replay request."""
        try:
            user_id = self.connection_users.get(connection_id)
            if not user_id:
                raise ValueError("User not found for connection")
            
            replay_request = ReplayRequest(
                client_id=connection_id,
                project_id=message_data.get("project_id"),
                session_id=message_data.get("session_id"),
                since_timestamp=datetime.fromisoformat(message_data["since"]) if message_data.get("since") else None,
                event_types=message_data.get("event_types"),
                max_events=message_data.get("max_events", 100),
                include_delivered=message_data.get("include_delivered", False)
            )
            
            # Get replay events from history manager
            replay_events = await self.history_manager.replay_events(replay_request)
            
            # Send replay events
            for event in replay_events:
                await self._send_to_connection(connection_id, event)
            
            # Send replay completion message
            completion_message = {
                "type": "replay_complete",
                "data": {
                    "events_replayed": len(replay_events),
                    "project_id": replay_request.project_id
                },
                "timestamp": datetime.utcnow().isoformat(),
                "correlation_id": str(uuid.uuid4())
            }
            
            await self._send_to_connection(connection_id, completion_message)
            
            self.metrics["reconnections"] += 1
            
        except Exception as e:
            logger.error("Replay request handling failed", 
                        connection_id=connection_id, error=str(e))
    
    async def _handle_user_preferences(self, connection_id: str, message_data: Dict[str, Any]) -> None:
        """Handle user preferences update."""
        try:
            user_id = self.connection_users.get(connection_id)
            if not user_id:
                raise ValueError("User not found for connection")
            
            preferences_data = message_data.get("preferences", {})
            
            # Create user preferences object
            preferences = UserPreferences(
                user_id=user_id,
                preferred_languages=preferences_data.get("preferred_languages", []),
                ignored_file_patterns=preferences_data.get("ignored_file_patterns", []),
                min_progress_updates=preferences_data.get("min_progress_updates", 10),
                high_impact_only=preferences_data.get("high_impact_only", False),
                notification_frequency=preferences_data.get("notification_frequency", "normal")
            )
            
            # Set preferences in event filter
            self.event_filter.set_user_preferences(user_id, preferences)
            
            # Send acknowledgment
            ack_message = {
                "type": "preferences_ack",
                "data": {
                    "status": "success",
                    "preferences": preferences_data
                },
                "timestamp": datetime.utcnow().isoformat(),
                "correlation_id": str(uuid.uuid4())
            }
            
            await self._send_to_connection(connection_id, ack_message)
            
        except Exception as e:
            logger.error("Preferences handling failed", 
                        connection_id=connection_id, error=str(e))
    
    async def _handle_ping(self, connection_id: str, message_data: Dict[str, Any]) -> None:
        """Handle ping request."""
        pong_message = {
            "type": "pong",
            "data": {
                "timestamp": datetime.utcnow().isoformat(),
                "connection_health": self.performance_manager.connection_pool.get_connection_health(connection_id)
            },
            "timestamp": datetime.utcnow().isoformat(),
            "correlation_id": message_data.get("correlation_id", str(uuid.uuid4()))
        }
        
        await self._send_to_connection(connection_id, pong_message)
    
    async def _handle_stats_request(self, connection_id: str) -> None:
        """Handle statistics request."""
        stats_message = {
            "type": "stats",
            "data": {
                "connection_stats": self.metrics,
                "performance_stats": self.performance_manager.get_performance_summary(),
                "event_filter_stats": self.event_filter.get_metrics(),
                "active_connections": len(self.active_connections)
            },
            "timestamp": datetime.utcnow().isoformat(),
            "correlation_id": str(uuid.uuid4())
        }
        
        await self._send_to_connection(connection_id, stats_message)
    
    async def _send_replay_events(self, connection_id: str, user_id: str) -> None:
        """Send recent events for client reconnection."""
        try:
            # Get connection metadata to determine relevant projects
            metadata = self.connection_metadata.get(connection_id, {})
            project_id = metadata.get("project_id")
            
            if project_id and self.history_manager:
                # Get recent events from history
                recent_events = await self.history_manager.get_recent_events(
                    project_id=project_id,
                    since=datetime.utcnow() - timedelta(minutes=5),  # Last 5 minutes
                    limit=20
                )
                
                for event_entry in recent_events:
                    event_message = {
                        "type": event_entry.event_type,
                        "data": event_entry.event_data,
                        "timestamp": event_entry.timestamp.isoformat(),
                        "correlation_id": event_entry.correlation_id,
                        "replay": True
                    }
                    
                    await self._send_to_connection(connection_id, event_message)
                
                logger.debug("Sent replay events", 
                           connection_id=connection_id, 
                           event_count=len(recent_events))
                
        except Exception as e:
            logger.debug("Failed to send replay events", 
                        connection_id=connection_id, error=str(e))
    
    async def _send_to_connection(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """Send message to specific connection with performance tracking."""
        if connection_id not in self.active_connections:
            return False
        
        try:
            websocket = self.active_connections[connection_id]
            
            # Apply performance optimizations
            message_json = json.dumps(message)
            message_size = len(message_json.encode('utf-8'))
            
            start_time = datetime.utcnow()
            await websocket.send_text(message_json)
            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Record performance metrics
            self.performance_manager.connection_pool.record_message_sent(
                connection_id, message_size, latency_ms, True
            )
            
            # Mark event as delivered in history
            if self.history_manager and message.get("correlation_id"):
                await self.history_manager.mark_event_delivered(
                    message["correlation_id"], connection_id
                )
            
            self.metrics["events_delivered"] += 1
            return True
            
        except Exception as e:
            logger.error("Failed to send message to connection", 
                        connection_id=connection_id, error=str(e))
            
            # Record failed send
            self.performance_manager.connection_pool.record_message_sent(
                connection_id, 0, 0, False
            )
            
            # Clean up failed connection
            await self._cleanup_connection(connection_id)
            return False
    
    async def _redis_event_listener(self) -> None:
        """Listen for Redis-distributed events."""
        try:
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe("project_index:websocket_events")
            
            logger.info("Started Redis event listener for WebSocket distribution")
            
            while True:
                try:
                    message = await pubsub.get_message(timeout=1.0)
                    if message and message["type"] == "message":
                        await self._handle_redis_event(message)
                        
                except Exception as e:
                    logger.error("Error processing Redis event", error=str(e))
                    await asyncio.sleep(1)
                    
        except Exception as e:
            logger.error("Redis event listener failed", error=str(e))
        finally:
            await pubsub.unsubscribe("project_index:websocket_events")
    
    async def _handle_redis_event(self, redis_message: Dict[str, Any]) -> None:
        """Handle event received from Redis."""
        try:
            data = json.loads(redis_message["data"].decode())
            event = data["event"]
            target_subscribers = data["subscribers"]
            
            # Send to target subscribers
            for connection_id in target_subscribers:
                if connection_id in self.active_connections:
                    # Apply filtering
                    user_id = self.connection_users.get(connection_id)
                    if user_id:
                        should_deliver = await self.event_filter.should_deliver_event(
                            ProjectIndexWebSocketEvent(
                                type=ProjectIndexEventType(event["type"]),
                                data=event["data"],
                                timestamp=datetime.fromisoformat(event["timestamp"]),
                                correlation_id=uuid.UUID(event["correlation_id"])
                            ),
                            user_id,
                            self.connection_metadata.get(connection_id)
                        )
                        
                        if should_deliver:
                            await self._send_to_connection(connection_id, event)
                        else:
                            self.metrics["events_filtered"] += 1
            
        except Exception as e:
            logger.error("Failed to handle Redis event", error=str(e))
    
    async def _cleanup_connection(self, connection_id: str) -> None:
        """Clean up connection resources."""
        try:
            # Remove from active connections
            self.active_connections.pop(connection_id, None)
            self.connection_users.pop(connection_id, None)
            self.connection_metadata.pop(connection_id, None)
            
            # Unsubscribe from events
            await self.event_publisher.unsubscribe_client(connection_id)
            
            # Remove from performance manager
            self.performance_manager.remove_connection(connection_id)
            
            logger.debug("Cleaned up connection", connection_id=connection_id)
            
        except Exception as e:
            logger.error("Connection cleanup failed", 
                        connection_id=connection_id, error=str(e))
    
    async def shutdown(self) -> None:
        """Shutdown WebSocket manager gracefully."""
        logger.info("Shutting down WebSocket manager")
        
        # Stop Redis subscriber
        if self.redis_subscriber_task:
            self.redis_subscriber_task.cancel()
            try:
                await self.redis_subscriber_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        for connection_id in list(self.active_connections.keys()):
            await self._cleanup_connection(connection_id)
        
        logger.info("WebSocket manager shutdown complete")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive WebSocket metrics."""
        return {
            "websocket_manager": self.metrics,
            "performance": self.performance_manager.get_performance_summary(),
            "event_filter": self.event_filter.get_metrics(),
            "active_connections": len(self.active_connections),
            "connection_details": {
                connection_id: {
                    "user_id": self.connection_users.get(connection_id),
                    "subscriptions": list(metadata.get("subscriptions", [])),
                    "connected_at": metadata.get("connected_at").isoformat() if metadata.get("connected_at") else None,
                    "last_activity": metadata.get("last_activity").isoformat() if metadata.get("last_activity") else None
                }
                for connection_id, metadata in self.connection_metadata.items()
            }
        }


# Global WebSocket manager instance
_websocket_manager: Optional[ProjectIndexWebSocketManager] = None


async def get_websocket_manager() -> ProjectIndexWebSocketManager:
    """Get or create the global WebSocket manager."""
    global _websocket_manager
    if _websocket_manager is None:
        _websocket_manager = ProjectIndexWebSocketManager()
        await _websocket_manager.initialize()
    return _websocket_manager


# FastAPI WebSocket endpoint
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str = "anonymous",  # In real implementation, extract from auth
    manager: ProjectIndexWebSocketManager = Depends(get_websocket_manager)
):
    """FastAPI WebSocket endpoint for project index events."""
    await manager.handle_connection(websocket, user_id)