"""
WebSocket Integration for Project Index Real-time Updates

Provides real-time analysis progress updates, file change notifications,
and dependency graph changes via WebSocket connections.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Set, Any, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import WebSocket, WebSocketDisconnect, Depends, HTTPException
from pydantic import BaseModel, ValidationError

from ..core.redis import get_redis_client, RedisClient

# Mock auth function for WebSocket
async def get_current_user_from_token(token: str):
    """Mock user from token - replace with actual implementation."""
    return "test_user"
from ..schemas.project_index import AnalysisProgress

logger = structlog.get_logger()


class WebSocketMessage(BaseModel):
    """Base WebSocket message schema."""
    type: str
    data: Dict
    timestamp: datetime = datetime.utcnow()
    correlation_id: str = str(uuid.uuid4())


class SubscriptionRequest(BaseModel):
    """WebSocket subscription request."""
    action: str  # "subscribe" or "unsubscribe"
    event_types: List[str]  # ["analysis_progress", "file_changes", "dependency_updates"]
    project_id: Optional[str] = None
    session_id: Optional[str] = None


class ConnectionManager:
    """Manages WebSocket connections and subscriptions."""
    
    def __init__(self):
        # Map of connection_id -> WebSocket
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Map of connection_id -> user_id for authentication
        self.connection_users: Dict[str, str] = {}
        
        # Map of project_id -> set of connection_ids subscribed to it
        self.project_subscriptions: Dict[str, Set[str]] = {}
        
        # Map of session_id -> set of connection_ids subscribed to it
        self.session_subscriptions: Dict[str, Set[str]] = {}
        
        # Map of connection_id -> set of event types they're subscribed to
        self.event_subscriptions: Dict[str, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket, connection_id: str, user_id: str):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        self.connection_users[connection_id] = user_id
        
        logger.info("WebSocket connection established", 
                   connection_id=connection_id, user_id=user_id)
    
    def disconnect(self, connection_id: str):
        """Clean up disconnected WebSocket."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        if connection_id in self.connection_users:
            del self.connection_users[connection_id]
        
        # Remove from all subscriptions
        for project_id, connections in self.project_subscriptions.items():
            connections.discard(connection_id)
        
        for session_id, connections in self.session_subscriptions.items():
            connections.discard(connection_id)
        
        if connection_id in self.event_subscriptions:
            del self.event_subscriptions[connection_id]
        
        logger.info("WebSocket connection cleaned up", connection_id=connection_id)
    
    def subscribe_to_project(self, connection_id: str, project_id: str):
        """Subscribe connection to project events."""
        if project_id not in self.project_subscriptions:
            self.project_subscriptions[project_id] = set()
        
        self.project_subscriptions[project_id].add(connection_id)
        
        logger.debug("Subscribed to project", 
                    connection_id=connection_id, project_id=project_id)
    
    def subscribe_to_session(self, connection_id: str, session_id: str):
        """Subscribe connection to analysis session events."""
        if session_id not in self.session_subscriptions:
            self.session_subscriptions[session_id] = set()
        
        self.session_subscriptions[session_id].add(connection_id)
        
        logger.debug("Subscribed to session", 
                    connection_id=connection_id, session_id=session_id)
    
    def subscribe_to_events(self, connection_id: str, event_types: List[str]):
        """Subscribe connection to specific event types."""
        if connection_id not in self.event_subscriptions:
            self.event_subscriptions[connection_id] = set()
        
        self.event_subscriptions[connection_id].update(event_types)
        
        logger.debug("Subscribed to events", 
                    connection_id=connection_id, event_types=event_types)
    
    def unsubscribe_from_project(self, connection_id: str, project_id: str):
        """Unsubscribe connection from project events."""
        if project_id in self.project_subscriptions:
            self.project_subscriptions[project_id].discard(connection_id)
    
    def unsubscribe_from_session(self, connection_id: str, session_id: str):
        """Unsubscribe connection from session events."""
        if session_id in self.session_subscriptions:
            self.session_subscriptions[session_id].discard(connection_id)
    
    async def send_to_connection(self, connection_id: str, message: WebSocketMessage):
        """Send message to specific connection."""
        if connection_id in self.active_connections:
            try:
                websocket = self.active_connections[connection_id]
                await websocket.send_text(message.model_dump_json())
            except Exception as e:
                logger.error("Failed to send WebSocket message", 
                           connection_id=connection_id, error=str(e))
                # Remove failed connection
                self.disconnect(connection_id)
    
    async def broadcast_to_project(self, project_id: str, message: WebSocketMessage):
        """Broadcast message to all connections subscribed to a project."""
        if project_id in self.project_subscriptions:
            connections = self.project_subscriptions[project_id].copy()
            for connection_id in connections:
                await self.send_to_connection(connection_id, message)
    
    async def broadcast_to_session(self, session_id: str, message: WebSocketMessage):
        """Broadcast message to all connections subscribed to a session."""
        if session_id in self.session_subscriptions:
            connections = self.session_subscriptions[session_id].copy()
            for connection_id in connections:
                await self.send_to_connection(connection_id, message)
    
    async def broadcast_analysis_progress(self, progress: AnalysisProgress):
        """Broadcast analysis progress to subscribed connections."""
        message = WebSocketMessage(
            type="analysis_progress",
            data=progress.model_dump()
        )
        
        # Send to project subscribers
        await self.broadcast_to_project(str(progress.project_id), message)
        
        # Send to session subscribers
        await self.broadcast_to_session(str(progress.session_id), message)
    
    async def broadcast_file_change(self, project_id: str, file_path: str, change_type: str):
        """Broadcast file change notification."""
        message = WebSocketMessage(
            type="file_change",
            data={
                "project_id": project_id,
                "file_path": file_path,
                "change_type": change_type,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        await self.broadcast_to_project(project_id, message)
    
    async def broadcast_dependency_update(self, project_id: str, update_type: str, details: Dict):
        """Broadcast dependency graph update."""
        message = WebSocketMessage(
            type="dependency_update",
            data={
                "project_id": project_id,
                "update_type": update_type,
                "details": details,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        await self.broadcast_to_project(project_id, message)


# Global connection manager instance
connection_manager = ConnectionManager()


class ProjectIndexWebSocketHandler:
    """Handles Project Index WebSocket connections and events."""
    
    def __init__(self, redis_client: RedisClient):
        self.redis = redis_client
        self.manager = connection_manager
    
    async def handle_connection(self, websocket: WebSocket, user_id: str):
        """Handle new WebSocket connection."""
        connection_id = str(uuid.uuid4())
        
        try:
            await self.manager.connect(websocket, connection_id, user_id)
            
            # Send welcome message
            welcome_message = WebSocketMessage(
                type="welcome",
                data={
                    "connection_id": connection_id,
                    "user_id": user_id,
                    "available_events": [
                        "project_index_updated",
                        "analysis_progress",
                        "dependency_changed",
                        "context_optimized",
                        "file_change", 
                        "dependency_update",
                        "project_status_change"
                    ]
                }
            )
            await self.manager.send_to_connection(connection_id, welcome_message)
            
            # Listen for messages
            while True:
                try:
                    data = await websocket.receive_text()
                    await self._handle_message(connection_id, data)
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error("WebSocket message handling error", 
                               connection_id=connection_id, error=str(e))
                    
                    error_message = WebSocketMessage(
                        type="error",
                        data={
                            "error": "MESSAGE_HANDLING_ERROR",
                            "message": "Failed to process message",
                            "details": str(e)
                        }
                    )
                    await self.manager.send_to_connection(connection_id, error_message)
        
        except Exception as e:
            logger.error("WebSocket connection error", 
                        connection_id=connection_id, error=str(e))
        
        finally:
            self.manager.disconnect(connection_id)
    
    async def _handle_message(self, connection_id: str, data: str):
        """Handle incoming WebSocket message."""
        try:
            message_data = json.loads(data)
            subscription = SubscriptionRequest(**message_data)
            
            if subscription.action == "subscribe":
                await self._handle_subscription(connection_id, subscription)
            elif subscription.action == "unsubscribe":
                await self._handle_unsubscription(connection_id, subscription)
            else:
                raise ValueError(f"Unknown action: {subscription.action}")
            
            # Send acknowledgment
            ack_message = WebSocketMessage(
                type="subscription_ack",
                data={
                    "action": subscription.action,
                    "event_types": subscription.event_types,
                    "project_id": subscription.project_id,
                    "session_id": subscription.session_id,
                    "status": "success"
                }
            )
            await self.manager.send_to_connection(connection_id, ack_message)
            
        except (json.JSONDecodeError, ValidationError) as e:
            error_message = WebSocketMessage(
                type="error",
                data={
                    "error": "INVALID_MESSAGE_FORMAT",
                    "message": "Invalid message format",
                    "details": str(e)
                }
            )
            await self.manager.send_to_connection(connection_id, error_message)
        
        except Exception as e:
            error_message = WebSocketMessage(
                type="error",
                data={
                    "error": "SUBSCRIPTION_ERROR",
                    "message": "Failed to process subscription",
                    "details": str(e)
                }
            )
            await self.manager.send_to_connection(connection_id, error_message)
    
    async def _handle_subscription(self, connection_id: str, subscription: SubscriptionRequest):
        """Handle subscription request."""
        # Subscribe to event types
        self.manager.subscribe_to_events(connection_id, subscription.event_types)
        
        # Subscribe to specific project if provided
        if subscription.project_id:
            self.manager.subscribe_to_project(connection_id, subscription.project_id)
        
        # Subscribe to specific session if provided
        if subscription.session_id:
            self.manager.subscribe_to_session(connection_id, subscription.session_id)
        
        logger.info("WebSocket subscription processed",
                   connection_id=connection_id,
                   event_types=subscription.event_types,
                   project_id=subscription.project_id,
                   session_id=subscription.session_id)
    
    async def _handle_unsubscription(self, connection_id: str, subscription: SubscriptionRequest):
        """Handle unsubscription request."""
        # Unsubscribe from specific project if provided
        if subscription.project_id:
            self.manager.unsubscribe_from_project(connection_id, subscription.project_id)
        
        # Unsubscribe from specific session if provided
        if subscription.session_id:
            self.manager.unsubscribe_from_session(connection_id, subscription.session_id)
        
        # Remove event type subscriptions
        if connection_id in self.manager.event_subscriptions:
            for event_type in subscription.event_types:
                self.manager.event_subscriptions[connection_id].discard(event_type)
        
        logger.info("WebSocket unsubscription processed",
                   connection_id=connection_id,
                   event_types=subscription.event_types,
                   project_id=subscription.project_id,
                   session_id=subscription.session_id)


# Global WebSocket handler instance
websocket_handler = None


async def get_websocket_handler(
    redis_client: RedisClient = Depends(get_redis_client)
) -> ProjectIndexWebSocketHandler:
    """Get WebSocket handler instance."""
    global websocket_handler
    if websocket_handler is None:
        websocket_handler = ProjectIndexWebSocketHandler(redis_client)
    return websocket_handler


@asynccontextmanager
async def websocket_auth_context(token: str):
    """Context manager for WebSocket authentication."""
    try:
        user_id = await get_current_user_from_token(token)
        yield user_id
    except Exception as e:
        logger.error("WebSocket authentication failed", token=token[:10], error=str(e))
        raise HTTPException(status_code=401, detail="Authentication failed")


# ================== EVENT PUBLISHING FUNCTIONS ==================

async def publish_analysis_progress(progress: AnalysisProgress):
    """Publish analysis progress event to WebSocket subscribers."""
    try:
        await connection_manager.broadcast_analysis_progress(progress)
    except Exception as e:
        logger.error("Failed to publish analysis progress", error=str(e))


async def publish_file_change(project_id: str, file_path: str, change_type: str):
    """Publish file change event to WebSocket subscribers."""
    try:
        await connection_manager.broadcast_file_change(project_id, file_path, change_type)
    except Exception as e:
        logger.error("Failed to publish file change", error=str(e))


async def publish_dependency_update(project_id: str, update_type: str, details: Dict):
    """Publish dependency update event to WebSocket subscribers."""
    try:
        await connection_manager.broadcast_dependency_update(project_id, update_type, details)
    except Exception as e:
        logger.error("Failed to publish dependency update", error=str(e))


# ================== REDIS INTEGRATION FOR DISTRIBUTED EVENTS ==================

class DistributedEventHandler:
    """Handles distributed events across multiple server instances via Redis."""
    
    def __init__(self, redis_client: RedisClient):
        self.redis = redis_client
        self.running = False
    
    async def start_listening(self):
        """Start listening for distributed events."""
        self.running = True
        
        # Subscribe to Redis channels
        channels = [
            "project_index:analysis_progress",
            "project_index:file_changes",
            "project_index:dependency_updates"
        ]
        
        try:
            pubsub = self.redis.pubsub()
            await pubsub.subscribe(*channels)
            
            logger.info("Started listening for distributed events", channels=channels)
            
            while self.running:
                try:
                    message = await pubsub.get_message(timeout=1.0)
                    if message and message["type"] == "message":
                        await self._handle_distributed_event(message)
                except Exception as e:
                    logger.error("Error processing distributed event", error=str(e))
                    await asyncio.sleep(1)
        
        except Exception as e:
            logger.error("Failed to listen for distributed events", error=str(e))
        
        finally:
            await pubsub.unsubscribe(*channels)
            logger.info("Stopped listening for distributed events")
    
    async def stop_listening(self):
        """Stop listening for distributed events."""
        self.running = False
    
    async def _handle_distributed_event(self, message):
        """Handle incoming distributed event."""
        try:
            channel = message["channel"].decode()
            data = json.loads(message["data"].decode())
            
            if channel == "project_index:analysis_progress":
                progress = AnalysisProgress(**data)
                await connection_manager.broadcast_analysis_progress(progress)
            
            elif channel == "project_index:file_changes":
                await connection_manager.broadcast_file_change(
                    data["project_id"],
                    data["file_path"],
                    data["change_type"]
                )
            
            elif channel == "project_index:dependency_updates":
                await connection_manager.broadcast_dependency_update(
                    data["project_id"],
                    data["update_type"],
                    data["details"]
                )
        
        except Exception as e:
            logger.error("Failed to handle distributed event", 
                        channel=message.get("channel"), error=str(e))


# Global distributed event handler
distributed_handler = None


async def get_distributed_handler(
    redis_client: RedisClient = Depends(get_redis_client)
) -> DistributedEventHandler:
    """Get distributed event handler instance."""
    global distributed_handler
    if distributed_handler is None:
        distributed_handler = DistributedEventHandler(redis_client)
    return distributed_handler


# ================== UTILITY FUNCTIONS ==================

def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager instance."""
    return connection_manager


async def broadcast_to_all_connections(message: WebSocketMessage):
    """Broadcast message to all active connections."""
    for connection_id in connection_manager.active_connections:
        await connection_manager.send_to_connection(connection_id, message)


async def get_active_connections_count() -> int:
    """Get count of active WebSocket connections."""
    return len(connection_manager.active_connections)


async def get_subscription_stats() -> Dict[str, Any]:
    """Get statistics about current subscriptions."""
    return {
        "active_connections": len(connection_manager.active_connections),
        "project_subscriptions": {
            project_id: len(connections) 
            for project_id, connections in connection_manager.project_subscriptions.items()
        },
        "session_subscriptions": {
            session_id: len(connections)
            for session_id, connections in connection_manager.session_subscriptions.items()
        },
        "event_type_subscriptions": {
            connection_id: list(events)
            for connection_id, events in connection_manager.event_subscriptions.items()
        }
    }