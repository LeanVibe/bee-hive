"""
WebSocket Integration for Real-time Agent Coordination

Real-time communication system for agent coordination events, collaboration
session management, conflict notifications, and progress synchronization
across the multi-agent development environment.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import structlog

from fastapi import WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from fastapi.routing import APIRouter
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import get_session
from ..core.redis import get_redis_client, RedisClient
from ..models.agent import Agent
from ..models.project_index import ProjectIndex
from .context_integration import AgentContextIntegration, get_agent_context_integration
from .collaboration_engine import CollaborativeDevelopmentEngine, get_collaborative_development_engine

logger = structlog.get_logger()


class WebSocketEventType(Enum):
    """Types of WebSocket events for agent coordination."""
    # Connection events
    AGENT_CONNECTED = "agent_connected"
    AGENT_DISCONNECTED = "agent_disconnected"
    
    # Context events
    CONTEXT_UPDATED = "context_updated"
    CONTEXT_REQUEST = "context_request"
    CONTEXT_RESPONSE = "context_response"
    
    # Task coordination events
    TASK_ASSIGNED = "task_assigned"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_HANDOFF = "task_handoff"
    
    # Collaboration events
    COLLABORATION_STARTED = "collaboration_started"
    COLLABORATION_PROGRESS = "collaboration_progress"
    COLLABORATION_CONFLICT = "collaboration_conflict"
    COLLABORATION_RESOLVED = "collaboration_resolved"
    COLLABORATION_ENDED = "collaboration_ended"
    
    # Knowledge sharing events
    KNOWLEDGE_SHARED = "knowledge_shared"
    KNOWLEDGE_ACCESSED = "knowledge_accessed"
    
    # Project events
    PROJECT_UPDATED = "project_updated"
    FILE_MODIFIED = "file_modified"
    DEPENDENCY_CHANGED = "dependency_changed"
    
    # System events
    HEALTH_ALERT = "health_alert"
    PERFORMANCE_WARNING = "performance_warning"
    SYSTEM_MAINTENANCE = "system_maintenance"


@dataclass
class WebSocketEvent:
    """WebSocket event structure."""
    event_id: str
    event_type: WebSocketEventType
    timestamp: datetime
    source_agent: Optional[str]
    target_agents: List[str]
    project_id: Optional[str]
    session_id: Optional[str]
    data: Dict[str, Any]
    priority: str = "normal"
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "source_agent": self.source_agent,
            "target_agents": self.target_agents,
            "project_id": self.project_id,
            "session_id": self.session_id,
            "data": self.data,
            "priority": self.priority,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }


class ConnectionManager:
    """Manages WebSocket connections for agents."""
    
    def __init__(self):
        # Active connections: agent_id -> WebSocket
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Connection metadata: agent_id -> connection info
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Subscription management: agent_id -> set of subscribed topics
        self.subscriptions: Dict[str, Set[str]] = {}
        
        # Project subscriptions: project_id -> set of agent_ids
        self.project_subscriptions: Dict[str, Set[str]] = {}
        
        # Session subscriptions: session_id -> set of agent_ids
        self.session_subscriptions: Dict[str, Set[str]] = {}
    
    async def connect(
        self,
        websocket: WebSocket,
        agent_id: str,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        
        # Store connection
        self.active_connections[agent_id] = websocket
        
        # Store metadata
        self.connection_metadata[agent_id] = {
            "connected_at": datetime.utcnow(),
            "project_id": project_id,
            "session_id": session_id,
            "last_ping": datetime.utcnow()
        }
        
        # Initialize subscriptions
        if agent_id not in self.subscriptions:
            self.subscriptions[agent_id] = set()
        
        # Auto-subscribe to relevant topics
        await self._setup_default_subscriptions(agent_id, project_id, session_id)
        
        logger.info(
            "Agent connected via WebSocket",
            agent_id=agent_id,
            project_id=project_id,
            session_id=session_id
        )
    
    async def disconnect(self, agent_id: str):
        """Handle agent disconnection."""
        # Remove connection
        if agent_id in self.active_connections:
            del self.active_connections[agent_id]
        
        # Clean up metadata
        metadata = self.connection_metadata.pop(agent_id, {})
        
        # Clean up subscriptions
        agent_subscriptions = self.subscriptions.pop(agent_id, set())
        
        # Remove from project subscriptions
        project_id = metadata.get("project_id")
        if project_id and project_id in self.project_subscriptions:
            self.project_subscriptions[project_id].discard(agent_id)
            if not self.project_subscriptions[project_id]:
                del self.project_subscriptions[project_id]
        
        # Remove from session subscriptions
        session_id = metadata.get("session_id")
        if session_id and session_id in self.session_subscriptions:
            self.session_subscriptions[session_id].discard(agent_id)
            if not self.session_subscriptions[session_id]:
                del self.session_subscriptions[session_id]
        
        logger.info(
            "Agent disconnected from WebSocket",
            agent_id=agent_id,
            subscriptions_cleaned=len(agent_subscriptions)
        )
    
    async def send_to_agent(self, agent_id: str, event: WebSocketEvent):
        """Send event to a specific agent."""
        if agent_id in self.active_connections:
            try:
                websocket = self.active_connections[agent_id]
                await websocket.send_text(json.dumps(event.to_dict()))
                return True
            except Exception as e:
                logger.warning(
                    "Failed to send WebSocket message",
                    agent_id=agent_id,
                    error=str(e)
                )
                # Connection may be stale, remove it
                await self.disconnect(agent_id)
                return False
        return False
    
    async def broadcast_to_agents(self, agent_ids: List[str], event: WebSocketEvent):
        """Broadcast event to multiple agents."""
        successful_sends = 0
        
        for agent_id in agent_ids:
            if await self.send_to_agent(agent_id, event):
                successful_sends += 1
        
        return successful_sends
    
    async def broadcast_to_project(self, project_id: str, event: WebSocketEvent):
        """Broadcast event to all agents subscribed to a project."""
        if project_id in self.project_subscriptions:
            agent_ids = list(self.project_subscriptions[project_id])
            return await self.broadcast_to_agents(agent_ids, event)
        return 0
    
    async def broadcast_to_session(self, session_id: str, event: WebSocketEvent):
        """Broadcast event to all agents in a collaboration session."""
        if session_id in self.session_subscriptions:
            agent_ids = list(self.session_subscriptions[session_id])
            return await self.broadcast_to_agents(agent_ids, event)
        return 0
    
    async def subscribe_agent(self, agent_id: str, topic: str):
        """Subscribe agent to a topic."""
        if agent_id not in self.subscriptions:
            self.subscriptions[agent_id] = set()
        
        self.subscriptions[agent_id].add(topic)
        
        # Handle special subscriptions
        if topic.startswith("project:"):
            project_id = topic.split(":", 1)[1]
            if project_id not in self.project_subscriptions:
                self.project_subscriptions[project_id] = set()
            self.project_subscriptions[project_id].add(agent_id)
        
        elif topic.startswith("session:"):
            session_id = topic.split(":", 1)[1]
            if session_id not in self.session_subscriptions:
                self.session_subscriptions[session_id] = set()
            self.session_subscriptions[session_id].add(agent_id)
    
    async def unsubscribe_agent(self, agent_id: str, topic: str):
        """Unsubscribe agent from a topic."""
        if agent_id in self.subscriptions:
            self.subscriptions[agent_id].discard(topic)
        
        # Handle special unsubscriptions
        if topic.startswith("project:"):
            project_id = topic.split(":", 1)[1]
            if project_id in self.project_subscriptions:
                self.project_subscriptions[project_id].discard(agent_id)
        
        elif topic.startswith("session:"):
            session_id = topic.split(":", 1)[1]
            if session_id in self.session_subscriptions:
                self.session_subscriptions[session_id].discard(agent_id)
    
    def get_connected_agents(self) -> List[str]:
        """Get list of currently connected agents."""
        return list(self.active_connections.keys())
    
    def get_connection_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get connection information for an agent."""
        return self.connection_metadata.get(agent_id)
    
    def get_project_agents(self, project_id: str) -> List[str]:
        """Get agents subscribed to a project."""
        return list(self.project_subscriptions.get(project_id, set()))
    
    def get_session_agents(self, session_id: str) -> List[str]:
        """Get agents in a collaboration session."""
        return list(self.session_subscriptions.get(session_id, set()))
    
    async def _setup_default_subscriptions(
        self,
        agent_id: str,
        project_id: Optional[str],
        session_id: Optional[str]
    ):
        """Set up default subscriptions for an agent."""
        # Always subscribe to agent-specific events
        await self.subscribe_agent(agent_id, f"agent:{agent_id}")
        
        # Subscribe to project events if project specified
        if project_id:
            await self.subscribe_agent(agent_id, f"project:{project_id}")
        
        # Subscribe to session events if session specified
        if session_id:
            await self.subscribe_agent(agent_id, f"session:{session_id}")
        
        # Subscribe to system events
        await self.subscribe_agent(agent_id, "system:alerts")
        await self.subscribe_agent(agent_id, "system:maintenance")


class WebSocketCoordinator:
    """
    Main coordinator for WebSocket-based agent communication.
    
    Handles real-time event broadcasting, subscription management,
    and coordination with the collaboration engine.
    """
    
    def __init__(
        self,
        redis_client: RedisClient,
        context_integration: AgentContextIntegration,
        collaboration_engine: CollaborativeDevelopmentEngine
    ):
        self.redis = redis_client
        self.context_integration = context_integration
        self.collaboration_engine = collaboration_engine
        
        # Connection manager
        self.connection_manager = ConnectionManager()
        
        # Event processing
        self.event_handlers: Dict[WebSocketEventType, List[Callable]] = {}
        self.event_queue_key = "websocket_events"
        self.event_processing = False
        
        # Configuration
        self.event_ttl = 3600  # 1 hour
        self.max_queue_size = 1000
        self.batch_size = 10
    
    async def start_event_processing(self):
        """Start background event processing."""
        if not self.event_processing:
            self.event_processing = True
            asyncio.create_task(self._process_events_loop())
            logger.info("WebSocket event processing started")
    
    async def stop_event_processing(self):
        """Stop background event processing."""
        self.event_processing = False
        logger.info("WebSocket event processing stopped")
    
    async def publish_event(self, event: WebSocketEvent):
        """Publish an event to the WebSocket system."""
        try:
            # Store event in Redis queue
            event_data = json.dumps(event.to_dict())
            await self.redis.lpush(self.event_queue_key, event_data)
            
            # Trim queue if too large
            await self.redis.ltrim(self.event_queue_key, 0, self.max_queue_size - 1)
            
            # Set TTL on queue
            await self.redis.expire(self.event_queue_key, self.event_ttl)
            
            logger.debug(
                "Event published to WebSocket queue",
                event_id=event.event_id,
                event_type=event.event_type.value
            )
            
        except Exception as e:
            logger.error(
                "Failed to publish WebSocket event",
                event_id=event.event_id,
                error=str(e)
            )
    
    async def handle_agent_context_update(
        self,
        agent_id: str,
        project_id: str,
        context_data: Dict[str, Any]
    ):
        """Handle agent context updates."""
        event = WebSocketEvent(
            event_id=str(uuid.uuid4()),
            event_type=WebSocketEventType.CONTEXT_UPDATED,
            timestamp=datetime.utcnow(),
            source_agent=agent_id,
            target_agents=[agent_id],
            project_id=project_id,
            session_id=None,
            data={
                "context_summary": {
                    "files_count": len(context_data.get("files", [])),
                    "dependencies_count": len(context_data.get("dependencies", [])),
                    "updated_at": datetime.utcnow().isoformat()
                }
            }
        )
        
        await self.publish_event(event)
    
    async def handle_task_assignment(
        self,
        task_id: str,
        agent_ids: List[str],
        project_id: Optional[str] = None,
        routing_metadata: Optional[Dict[str, Any]] = None
    ):
        """Handle task assignment events."""
        event = WebSocketEvent(
            event_id=str(uuid.uuid4()),
            event_type=WebSocketEventType.TASK_ASSIGNED,
            timestamp=datetime.utcnow(),
            source_agent=None,
            target_agents=agent_ids,
            project_id=project_id,
            session_id=None,
            data={
                "task_id": task_id,
                "assigned_agents": agent_ids,
                "routing_metadata": routing_metadata or {},
                "assignment_time": datetime.utcnow().isoformat()
            },
            priority="high"
        )
        
        await self.publish_event(event)
    
    async def handle_task_progress(
        self,
        task_id: str,
        agent_id: str,
        progress_data: Dict[str, Any],
        project_id: Optional[str] = None
    ):
        """Handle task progress updates."""
        event = WebSocketEvent(
            event_id=str(uuid.uuid4()),
            event_type=WebSocketEventType.TASK_PROGRESS,
            timestamp=datetime.utcnow(),
            source_agent=agent_id,
            target_agents=[],  # Will be determined by subscriptions
            project_id=project_id,
            session_id=None,
            data={
                "task_id": task_id,
                "progress": progress_data,
                "updated_by": agent_id,
                "update_time": datetime.utcnow().isoformat()
            }
        )
        
        await self.publish_event(event)
    
    async def handle_collaboration_event(
        self,
        session_id: str,
        event_type: WebSocketEventType,
        data: Dict[str, Any],
        source_agent: Optional[str] = None
    ):
        """Handle collaboration session events."""
        # Get session participants
        session_agents = self.connection_manager.get_session_agents(session_id)
        
        event = WebSocketEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.utcnow(),
            source_agent=source_agent,
            target_agents=session_agents,
            project_id=data.get("project_id"),
            session_id=session_id,
            data=data,
            priority="high" if event_type == WebSocketEventType.COLLABORATION_CONFLICT else "normal"
        )
        
        await self.publish_event(event)
    
    async def handle_conflict_notification(
        self,
        conflict_id: str,
        involved_agents: List[str],
        conflict_data: Dict[str, Any],
        project_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Handle collaboration conflict notifications."""
        event = WebSocketEvent(
            event_id=str(uuid.uuid4()),
            event_type=WebSocketEventType.COLLABORATION_CONFLICT,
            timestamp=datetime.utcnow(),
            source_agent=None,
            target_agents=involved_agents,
            project_id=project_id,
            session_id=session_id,
            data={
                "conflict_id": conflict_id,
                "conflict_type": conflict_data.get("conflict_type"),
                "severity": conflict_data.get("severity"),
                "description": conflict_data.get("description"),
                "affected_resources": conflict_data.get("affected_resources", []),
                "resolution_suggestions": conflict_data.get("resolution_suggestions", [])
            },
            priority="critical"
        )
        
        await self.publish_event(event)
    
    async def handle_knowledge_sharing(
        self,
        share_id: str,
        source_agent: str,
        target_agents: List[str],
        knowledge_data: Dict[str, Any],
        project_id: Optional[str] = None
    ):
        """Handle knowledge sharing events."""
        event = WebSocketEvent(
            event_id=str(uuid.uuid4()),
            event_type=WebSocketEventType.KNOWLEDGE_SHARED,
            timestamp=datetime.utcnow(),
            source_agent=source_agent,
            target_agents=target_agents,
            project_id=project_id,
            session_id=None,
            data={
                "share_id": share_id,
                "knowledge_type": knowledge_data.get("knowledge_type"),
                "content_preview": str(knowledge_data.get("content", ""))[:200],
                "shared_by": source_agent,
                "share_time": datetime.utcnow().isoformat()
            }
        )
        
        await self.publish_event(event)
    
    async def handle_project_update(
        self,
        project_id: str,
        update_type: str,
        update_data: Dict[str, Any]
    ):
        """Handle project update events."""
        event = WebSocketEvent(
            event_id=str(uuid.uuid4()),
            event_type=WebSocketEventType.PROJECT_UPDATED,
            timestamp=datetime.utcnow(),
            source_agent=None,
            target_agents=[],  # Will broadcast to project subscribers
            project_id=project_id,
            session_id=None,
            data={
                "update_type": update_type,
                "update_data": update_data,
                "update_time": datetime.utcnow().isoformat()
            }
        )
        
        await self.publish_event(event)
    
    async def handle_health_alert(
        self,
        alert_type: str,
        alert_data: Dict[str, Any],
        project_id: Optional[str] = None,
        severity: str = "medium"
    ):
        """Handle system health alerts."""
        # Determine target agents based on project or all connected
        if project_id:
            target_agents = self.connection_manager.get_project_agents(project_id)
        else:
            target_agents = self.connection_manager.get_connected_agents()
        
        event = WebSocketEvent(
            event_id=str(uuid.uuid4()),
            event_type=WebSocketEventType.HEALTH_ALERT,
            timestamp=datetime.utcnow(),
            source_agent=None,
            target_agents=target_agents,
            project_id=project_id,
            session_id=None,
            data={
                "alert_type": alert_type,
                "severity": severity,
                "alert_data": alert_data,
                "alert_time": datetime.utcnow().isoformat()
            },
            priority="critical" if severity == "critical" else "high"
        )
        
        await self.publish_event(event)
    
    def register_event_handler(
        self,
        event_type: WebSocketEventType,
        handler: Callable[[WebSocketEvent], None]
    ):
        """Register a custom event handler."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
    
    async def _process_events_loop(self):
        """Background loop for processing WebSocket events."""
        while self.event_processing:
            try:
                # Get events from queue
                events = await self._get_events_batch()
                
                if events:
                    await self._process_events_batch(events)
                else:
                    # No events, wait a bit
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error("Error in WebSocket event processing loop", error=str(e))
                await asyncio.sleep(1)
    
    async def _get_events_batch(self) -> List[WebSocketEvent]:
        """Get a batch of events from the queue."""
        try:
            # Get events from Redis queue
            event_data_list = await self.redis.lrange(
                self.event_queue_key, 0, self.batch_size - 1
            )
            
            if event_data_list:
                # Remove processed events from queue
                await self.redis.ltrim(
                    self.event_queue_key, len(event_data_list), -1
                )
            
            # Parse events
            events = []
            for event_data in event_data_list:
                try:
                    event_dict = json.loads(event_data)
                    event = self._dict_to_event(event_dict)
                    events.append(event)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning("Failed to parse WebSocket event", error=str(e))
                    continue
            
            return events
            
        except Exception as e:
            logger.error("Failed to get events batch", error=str(e))
            return []
    
    async def _process_events_batch(self, events: List[WebSocketEvent]):
        """Process a batch of WebSocket events."""
        for event in events:
            try:
                await self._process_event(event)
            except Exception as e:
                logger.error(
                    "Failed to process WebSocket event",
                    event_id=event.event_id,
                    error=str(e)
                )
    
    async def _process_event(self, event: WebSocketEvent):
        """Process a single WebSocket event."""
        # Check if event has expired
        if event.expires_at and datetime.utcnow() > event.expires_at:
            logger.debug("Skipping expired WebSocket event", event_id=event.event_id)
            return
        
        # Determine target agents
        target_agents = event.target_agents.copy()
        
        # Add agents based on subscriptions
        if event.project_id:
            project_agents = self.connection_manager.get_project_agents(event.project_id)
            target_agents.extend(project_agents)
        
        if event.session_id:
            session_agents = self.connection_manager.get_session_agents(event.session_id)
            target_agents.extend(session_agents)
        
        # Remove duplicates
        target_agents = list(set(target_agents))
        
        # Send event to target agents
        successful_sends = await self.connection_manager.broadcast_to_agents(
            target_agents, event
        )
        
        # Run custom event handlers
        if event.event_type in self.event_handlers:
            for handler in self.event_handlers[event.event_type]:
                try:
                    await handler(event)
                except Exception as e:
                    logger.warning(
                        "Event handler failed",
                        event_type=event.event_type.value,
                        error=str(e)
                    )
        
        logger.debug(
            "WebSocket event processed",
            event_id=event.event_id,
            event_type=event.event_type.value,
            target_agents_count=len(target_agents),
            successful_sends=successful_sends
        )
    
    def _dict_to_event(self, event_dict: Dict[str, Any]) -> WebSocketEvent:
        """Convert dictionary to WebSocketEvent."""
        return WebSocketEvent(
            event_id=event_dict["event_id"],
            event_type=WebSocketEventType(event_dict["event_type"]),
            timestamp=datetime.fromisoformat(event_dict["timestamp"]),
            source_agent=event_dict.get("source_agent"),
            target_agents=event_dict.get("target_agents", []),
            project_id=event_dict.get("project_id"),
            session_id=event_dict.get("session_id"),
            data=event_dict.get("data", {}),
            priority=event_dict.get("priority", "normal"),
            expires_at=datetime.fromisoformat(event_dict["expires_at"]) if event_dict.get("expires_at") else None
        )


# Global coordinator instance
websocket_coordinator: Optional[WebSocketCoordinator] = None


# WebSocket router
websocket_router = APIRouter()


@websocket_router.websocket("/ws/agent/{agent_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    agent_id: str,
    project_id: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for agent real-time coordination.
    
    Provides real-time communication channel for agents to receive
    coordination events, collaboration updates, and system notifications.
    """
    global websocket_coordinator
    
    # Initialize coordinator if not already done
    if websocket_coordinator is None:
        websocket_coordinator = WebSocketCoordinator(
            redis_client, context_integration, collaboration_engine
        )
        await websocket_coordinator.start_event_processing()
    
    # Accept connection
    await websocket_coordinator.connection_manager.connect(
        websocket, agent_id, project_id, session_id
    )
    
    try:
        # Send welcome message
        welcome_event = WebSocketEvent(
            event_id=str(uuid.uuid4()),
            event_type=WebSocketEventType.AGENT_CONNECTED,
            timestamp=datetime.utcnow(),
            source_agent=None,
            target_agents=[agent_id],
            project_id=project_id,
            session_id=session_id,
            data={
                "message": "Connected to agent coordination system",
                "agent_id": agent_id,
                "capabilities": ["context_updates", "task_coordination", "collaboration", "knowledge_sharing"]
            }
        )
        
        await websocket_coordinator.connection_manager.send_to_agent(agent_id, welcome_event)
        
        # Listen for incoming messages
        while True:
            try:
                # Receive message from agent
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                await _handle_agent_message(agent_id, message, websocket_coordinator)
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                # Send error message
                error_event = WebSocketEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=WebSocketEventType.SYSTEM_MAINTENANCE,
                    timestamp=datetime.utcnow(),
                    source_agent=None,
                    target_agents=[agent_id],
                    project_id=project_id,
                    session_id=session_id,
                    data={"error": "Invalid JSON message format"},
                    priority="low"
                )
                await websocket_coordinator.connection_manager.send_to_agent(agent_id, error_event)
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WebSocket connection error", agent_id=agent_id, error=str(e))
    finally:
        # Handle disconnection
        await websocket_coordinator.connection_manager.disconnect(agent_id)
        
        # Send disconnection event
        disconnect_event = WebSocketEvent(
            event_id=str(uuid.uuid4()),
            event_type=WebSocketEventType.AGENT_DISCONNECTED,
            timestamp=datetime.utcnow(),
            source_agent=agent_id,
            target_agents=[],
            project_id=project_id,
            session_id=session_id,
            data={
                "agent_id": agent_id,
                "disconnected_at": datetime.utcnow().isoformat()
            }
        )
        
        if websocket_coordinator:
            await websocket_coordinator.publish_event(disconnect_event)


async def _handle_agent_message(
    agent_id: str,
    message: Dict[str, Any],
    coordinator: WebSocketCoordinator
):
    """Handle incoming message from agent."""
    message_type = message.get("type")
    
    if message_type == "ping":
        # Handle ping/keepalive
        pong_event = WebSocketEvent(
            event_id=str(uuid.uuid4()),
            event_type=WebSocketEventType.SYSTEM_MAINTENANCE,
            timestamp=datetime.utcnow(),
            source_agent=None,
            target_agents=[agent_id],
            project_id=None,
            session_id=None,
            data={"type": "pong", "timestamp": datetime.utcnow().isoformat()}
        )
        await coordinator.connection_manager.send_to_agent(agent_id, pong_event)
    
    elif message_type == "subscribe":
        # Handle subscription request
        topic = message.get("topic")
        if topic:
            await coordinator.connection_manager.subscribe_agent(agent_id, topic)
    
    elif message_type == "unsubscribe":
        # Handle unsubscription request
        topic = message.get("topic")
        if topic:
            await coordinator.connection_manager.unsubscribe_agent(agent_id, topic)
    
    elif message_type == "task_progress":
        # Handle task progress update
        task_id = message.get("task_id")
        progress_data = message.get("progress", {})
        project_id = message.get("project_id")
        
        if task_id:
            await coordinator.handle_task_progress(task_id, agent_id, progress_data, project_id)
    
    elif message_type == "knowledge_share":
        # Handle knowledge sharing
        share_data = message.get("share_data", {})
        target_agents = message.get("target_agents", [])
        project_id = message.get("project_id")
        
        if share_data and target_agents:
            await coordinator.handle_knowledge_sharing(
                str(uuid.uuid4()),
                agent_id,
                target_agents,
                share_data,
                project_id
            )
    
    else:
        logger.warning("Unknown message type from agent", agent_id=agent_id, message_type=message_type)


# Utility functions for external integration
async def get_websocket_coordinator() -> Optional[WebSocketCoordinator]:
    """Get the global WebSocket coordinator instance."""
    return websocket_coordinator


async def publish_context_update(
    agent_id: str,
    project_id: str,
    context_data: Dict[str, Any]
):
    """Publish a context update event."""
    coordinator = await get_websocket_coordinator()
    if coordinator:
        await coordinator.handle_agent_context_update(agent_id, project_id, context_data)


async def publish_task_assignment(
    task_id: str,
    agent_ids: List[str],
    project_id: Optional[str] = None,
    routing_metadata: Optional[Dict[str, Any]] = None
):
    """Publish a task assignment event."""
    coordinator = await get_websocket_coordinator()
    if coordinator:
        await coordinator.handle_task_assignment(task_id, agent_ids, project_id, routing_metadata)


async def publish_collaboration_event(
    session_id: str,
    event_type: str,
    data: Dict[str, Any],
    source_agent: Optional[str] = None
):
    """Publish a collaboration event."""
    coordinator = await get_websocket_coordinator()
    if coordinator and event_type in [e.value for e in WebSocketEventType]:
        await coordinator.handle_collaboration_event(
            session_id,
            WebSocketEventType(event_type),
            data,
            source_agent
        )


async def publish_conflict_notification(
    conflict_id: str,
    involved_agents: List[str],
    conflict_data: Dict[str, Any],
    project_id: Optional[str] = None,
    session_id: Optional[str] = None
):
    """Publish a conflict notification."""
    coordinator = await get_websocket_coordinator()
    if coordinator:
        await coordinator.handle_conflict_notification(
            conflict_id, involved_agents, conflict_data, project_id, session_id
        )


async def publish_health_alert(
    alert_type: str,
    alert_data: Dict[str, Any],
    project_id: Optional[str] = None,
    severity: str = "medium"
):
    """Publish a health alert."""
    coordinator = await get_websocket_coordinator()
    if coordinator:
        await coordinator.handle_health_alert(alert_type, alert_data, project_id, severity)


async def get_connection_stats() -> Dict[str, Any]:
    """Get WebSocket connection statistics."""
    coordinator = await get_websocket_coordinator()
    if not coordinator:
        return {"error": "WebSocket coordinator not initialized"}
    
    connected_agents = coordinator.connection_manager.get_connected_agents()
    
    return {
        "connected_agents_count": len(connected_agents),
        "connected_agents": connected_agents,
        "project_subscriptions": {
            project_id: list(agents) 
            for project_id, agents in coordinator.connection_manager.project_subscriptions.items()
        },
        "session_subscriptions": {
            session_id: list(agents)
            for session_id, agents in coordinator.connection_manager.session_subscriptions.items()
        },
        "total_subscriptions": sum(
            len(subs) for subs in coordinator.connection_manager.subscriptions.values()
        )
    }


# Factory function for dependency injection
async def get_websocket_coordinator_instance(
    redis_client: RedisClient = None,
    context_integration: AgentContextIntegration = None,
    collaboration_engine: CollaborativeDevelopmentEngine = None
) -> WebSocketCoordinator:
    """Factory function to create WebSocketCoordinator instance."""
    if redis_client is None:
        redis_client = await get_redis_client()
    if context_integration is None:
        context_integration = await get_agent_context_integration()
    if collaboration_engine is None:
        collaboration_engine = await get_collaborative_development_engine()
    
    return WebSocketCoordinator(redis_client, context_integration, collaboration_engine)