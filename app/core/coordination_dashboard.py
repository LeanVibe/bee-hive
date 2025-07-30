"""
Coordination Dashboard Core System for LeanVibe Agent Hive 2.0

Provides real-time visual monitoring of multi-agent coordination patterns with
comprehensive graph-based representation, session management, and event filtering.

Key Features:
- Visual agent graph with real-time updates
- Session-based color coding and organization
- Advanced event filtering and streaming
- Chat transcript analysis and history
- Integration with enhanced lifecycle hooks system
"""

import asyncio
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from uuid import UUID, uuid4
from dataclasses import dataclass, asdict
from enum import Enum

import structlog
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from .enhanced_lifecycle_hooks import EnhancedEventType, LifecycleEventData
from .redis import get_message_broker
from ..models.agent import Agent
from ..models.task import Task
from ..models.context import Context

logger = structlog.get_logger()


class AgentNodeType(Enum):
    """Types of nodes in the agent graph."""
    AGENT = "agent"
    TOOL = "tool"
    SESSION = "session"
    CONTEXT = "context"


class AgentNodeStatus(Enum):
    """Status states for agent nodes."""
    ACTIVE = "active"
    SLEEPING = "sleeping"
    ERROR = "error"
    COMPLETED = "completed"
    IDLE = "idle"


class AgentEdgeType(Enum):
    """Types of edges in the agent graph."""
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    DATA_FLOW = "data_flow"
    CONTEXT_SHARE = "context_share"
    COORDINATION = "coordination"


@dataclass
class AgentGraphNode:
    """Node in the agent coordination graph."""
    id: str
    label: str
    type: AgentNodeType
    status: AgentNodeStatus
    position: Dict[str, float]  # x, y coordinates
    metadata: Dict[str, Any]
    
    # Visual properties
    size: float = 1.0
    color: str = "#4A90E2"
    shape: str = "circle"
    
    # Timing information
    created_at: datetime = None
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'type': self.type.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }


@dataclass
class AgentGraphEdge:
    """Edge in the agent coordination graph."""
    id: str
    source: str
    target: str
    type: AgentEdgeType
    weight: float  # interaction frequency/strength
    timestamp: datetime
    metadata: Dict[str, Any]
    
    # Visual properties
    color: str = "#888888"
    width: float = 1.0
    style: str = "solid"  # solid, dashed, dotted
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'type': self.type.value,
            'timestamp': self.timestamp.isoformat()
        }


class EventFilter(BaseModel):
    """Filter configuration for coordination events."""
    session_ids: List[str] = Field(default_factory=list)
    agent_types: List[str] = Field(default_factory=list)
    event_types: List[str] = Field(default_factory=list)
    node_types: List[str] = Field(default_factory=list)
    time_range: Optional[Dict[str, str]] = None  # start, end ISO strings
    severity_levels: List[str] = Field(default_factory=list)
    include_system_events: bool = True
    max_events: int = 1000


class SessionColorManager:
    """Manages unique color assignment for development sessions."""
    
    def __init__(self):
        # Predefined color palette for consistent session identification
        self.color_palette = [
            "#4A90E2",  # Blue
            "#50C878",  # Green  
            "#FF6B6B",  # Red
            "#FFD93D",  # Yellow
            "#9B59B6",  # Purple
            "#FF8C42",  # Orange
            "#26D0CE",  # Cyan
            "#FF69B4",  # Pink
            "#8FBC8F",  # Dark Sea Green
            "#DDA0DD",  # Plum
            "#20B2AA",  # Light Sea Green
            "#F0E68C",  # Khaki
        ]
        self.session_colors: Dict[str, str] = {}
        self.color_index = 0
    
    def get_session_color(self, session_id: str) -> str:
        """Get consistent color for a session ID."""
        if session_id not in self.session_colors:
            # Use hash for deterministic color assignment
            hash_value = int(hashlib.sha256(session_id.encode()).hexdigest()[:8], 16)
            color_index = hash_value % len(self.color_palette)
            self.session_colors[session_id] = self.color_palette[color_index]
        
        return self.session_colors[session_id]
    
    def get_event_color_map(self, session_id: str) -> Dict[str, str]:
        """Get color mapping for different event types within a session."""
        base_color = self.get_session_color(session_id)
        
        # Generate variations of the base color for different event types
        return {
            "tool_execution": base_color,
            "context_creation": self._lighten_color(base_color, 0.3),
            "agent_communication": self._darken_color(base_color, 0.2),
            "system_event": self._saturate_color(base_color, 0.5),
            "error_event": "#FF4444",  # Always red for errors
            "success_event": "#44FF44",  # Always green for success
        }
    
    def _lighten_color(self, hex_color: str, factor: float) -> str:
        """Lighten a hex color by a factor (0.0 to 1.0)."""
        # Simple implementation - in production would use proper color manipulation
        return hex_color  # Placeholder
    
    def _darken_color(self, hex_color: str, factor: float) -> str:
        """Darken a hex color by a factor (0.0 to 1.0)."""
        return hex_color  # Placeholder
    
    def _saturate_color(self, hex_color: str, factor: float) -> str:
        """Adjust saturation of a hex color."""
        return hex_color  # Placeholder


class AgentCommunicationEvent(BaseModel):
    """Represents communication between agents."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    source_agent_id: str
    target_agent_id: Optional[str] = None  # None for broadcast
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    context_shared: bool = False
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CoordinationDashboard:
    """
    Core coordination dashboard system for real-time agent monitoring.
    
    Provides visual graph representation of agent interactions, session management,
    and comprehensive event filtering for multi-agent system coordination.
    """
    
    def __init__(self):
        self.nodes: Dict[str, AgentGraphNode] = {}
        self.edges: Dict[str, AgentGraphEdge] = {}
        self.active_websockets: Dict[str, WebSocket] = {}
        self.session_color_manager = SessionColorManager()
        
        # Event processing and filtering
        self.event_filters: Dict[str, EventFilter] = {}
        self.communication_history: List[AgentCommunicationEvent] = []
        
        # Performance optimization
        self.max_nodes = 500
        self.max_edges = 2000
        self.max_history = 10000
        
        # Graph layout and visualization state
        self.graph_layout_cache: Dict[str, Dict[str, Any]] = {}
        self.last_layout_update = datetime.utcnow()
    
    async def register_websocket(
        self, 
        websocket: WebSocket, 
        session_id: str,
        event_filter: Optional[EventFilter] = None
    ) -> str:
        """Register a WebSocket connection for real-time updates."""
        connection_id = str(uuid4())
        
        await websocket.accept()
        self.active_websockets[connection_id] = websocket
        
        if event_filter:
            self.event_filters[connection_id] = event_filter
        
        logger.info(
            "WebSocket registered for coordination dashboard",
            connection_id=connection_id,
            session_id=session_id,
            total_connections=len(self.active_websockets)
        )
        
        # Send initial graph state
        await self._send_initial_graph_state(websocket, session_id, event_filter)
        
        return connection_id
    
    async def unregister_websocket(self, connection_id: str) -> None:
        """Unregister a WebSocket connection."""
        if connection_id in self.active_websockets:
            del self.active_websockets[connection_id]
            
        if connection_id in self.event_filters:
            del self.event_filters[connection_id]
        
        logger.info(
            "WebSocket unregistered from coordination dashboard",
            connection_id=connection_id,
            remaining_connections=len(self.active_websockets)
        )
    
    async def process_lifecycle_event(self, event: LifecycleEventData) -> None:
        """Process enhanced lifecycle events for graph updates."""
        try:
            # Update graph based on event type
            if event.event_type in [EnhancedEventType.AGENT_LIFECYCLE_START, EnhancedEventType.AGENT_LIFECYCLE_RESUME]:
                await self._handle_agent_activation(event)
            elif event.event_type == EnhancedEventType.AGENT_LIFECYCLE_PAUSE:
                await self._handle_agent_sleep(event)
            elif event.event_type in [EnhancedEventType.PRE_TOOL_USE, EnhancedEventType.POST_TOOL_USE]:
                await self._handle_tool_interaction(event)
            elif event.event_type in [EnhancedEventType.TASK_ASSIGNMENT, EnhancedEventType.AGENT_COORDINATION]:
                await self._handle_context_creation(event)
            elif event.event_type in [EnhancedEventType.ERROR_PATTERN_DETECTED, EnhancedEventType.PERFORMANCE_DEGRADATION]:
                await self._handle_error_event(event)
            
            # Broadcast updates to connected WebSockets
            await self._broadcast_graph_update(event)
            
        except Exception as e:
            logger.error(
                "Error processing lifecycle event for coordination dashboard",
                agent_id=event.agent_id,
                session_id=event.session_id,
                event_type=event.event_type.value,
                error=str(e)
            )
    
    async def _handle_agent_activation(self, event: LifecycleEventData) -> None:
        """Handle agent activation events."""
        agent_id = event.agent_id
        session_id = event.session_id
        
        if not agent_id:
            return
        
        # Create or update agent node
        node_id = f"agent_{agent_id}"
        
        if node_id in self.nodes:
            # Update existing node
            self.nodes[node_id].status = AgentNodeStatus.ACTIVE
            self.nodes[node_id].last_updated = datetime.utcnow()
            self.nodes[node_id].metadata.update({
                "agent_id": agent_id,
                "session_id": session_id,
                "last_activity": event.timestamp,
                "uptime": self._calculate_uptime(self.nodes[node_id].created_at),
                **event.payload
            })
        else:
            # Create new agent node
            position = await self._calculate_node_position(node_id, AgentNodeType.AGENT)
            color = self.session_color_manager.get_session_color(session_id)
            
            self.nodes[node_id] = AgentGraphNode(
                id=node_id,
                label=f"Agent {agent_id[:8]}",
                type=AgentNodeType.AGENT,
                status=AgentNodeStatus.ACTIVE,
                position=position,
                color=color,
                metadata={
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "agent_type": event.payload.get("agent_type", "unknown"),
                    "current_task": event.payload.get("current_task"),
                    "uptime": 0
                }
            )
        
        # Record communication event
        comm_event = AgentCommunicationEvent(
            session_id=session_id,
            source_agent_id=agent_id,
            message_type="agent_activation",
            content={"event_type": event.event_type.value, "payload": event.payload}
        )
        
        await self._add_communication_event(comm_event)
    
    async def _handle_agent_sleep(self, event: LifecycleEventData) -> None:
        """Handle agent sleep events."""
        agent_id = event.agent_id
        if not agent_id:
            return
        
        node_id = f"agent_{agent_id}"
        if node_id in self.nodes:
            self.nodes[node_id].status = AgentNodeStatus.SLEEPING
            self.nodes[node_id].last_updated = datetime.utcnow()
            self.nodes[node_id].color = "#888888"  # Gray for sleeping
            self.nodes[node_id].metadata.update({
                "sleep_reason": event.payload.get("sleep_reason"),
                "context_usage": event.payload.get("context_usage")
            })
    
    async def _handle_tool_interaction(self, event: LifecycleEventData) -> None:
        """Handle tool interaction events."""
        agent_id = event.agent_id
        tool_id = event.payload.get("tool_id")
        
        if not agent_id or not tool_id:
            return
        
        agent_node_id = f"agent_{agent_id}"
        tool_node_id = f"tool_{tool_id}"
        
        # Create tool node if it doesn't exist
        if tool_node_id not in self.nodes:
            position = await self._calculate_node_position(tool_node_id, AgentNodeType.TOOL)
            
            self.nodes[tool_node_id] = AgentGraphNode(
                id=tool_node_id,
                label=tool_id,
                type=AgentNodeType.TOOL,
                status=AgentNodeStatus.ACTIVE,
                position=position,
                color="#FFA500",  # Orange for tools
                shape="square",
                size=0.8,
                metadata={
                    "tool_id": tool_id,
                    "tool_type": event.payload.get("tool_type"),
                    "usage_count": 1
                }
            )
        else:
            # Update usage count
            self.nodes[tool_node_id].metadata["usage_count"] += 1
        
        # Create or update edge between agent and tool
        edge_id = f"{agent_node_id}_{tool_node_id}"
        
        if edge_id in self.edges:
            # Strengthen existing edge
            self.edges[edge_id].weight += 1
            self.edges[edge_id].timestamp = datetime.utcnow()
            self.edges[edge_id].width = min(5.0, 1.0 + self.edges[edge_id].weight * 0.1)
        else:
            # Create new edge
            self.edges[edge_id] = AgentGraphEdge(
                id=edge_id,
                source=agent_node_id,
                target=tool_node_id,
                type=AgentEdgeType.TOOL_CALL,
                weight=1.0,
                timestamp=datetime.utcnow(),
                color="#FFA500",
                metadata={
                    "tool_call_count": 1,
                    "last_success": event.event_type == EnhancedEventType.POST_TOOL_USE and 
                                   event.payload.get("success", False)
                }
            )
    
    async def _handle_context_creation(self, event: LifecycleEventData) -> None:
        """Handle context creation and sharing events."""
        context_id = event.payload.get("context_id")
        agent_id = event.agent_id
        
        if not context_id or not agent_id:
            return
        
        context_node_id = f"context_{context_id}"
        agent_node_id = f"agent_{agent_id}"
        
        # Create context node
        if context_node_id not in self.nodes:
            position = await self._calculate_node_position(context_node_id, AgentNodeType.CONTEXT)
            
            self.nodes[context_node_id] = AgentGraphNode(
                id=context_node_id,
                label=f"Context {context_id[:8]}",
                type=AgentNodeType.CONTEXT,
                status=AgentNodeStatus.ACTIVE,
                position=position,
                color="#8A2BE2",  # Blue Violet for context
                shape="diamond",
                size=0.6,
                metadata={
                    "context_id": context_id,
                    "context_type": event.payload.get("context_type"),
                    "sharing_agents": [agent_id]
                }
            )
        else:
            # Add agent to sharing list
            sharing_agents = self.nodes[context_node_id].metadata.get("sharing_agents", [])
            if agent_id not in sharing_agents:
                sharing_agents.append(agent_id)
                self.nodes[context_node_id].metadata["sharing_agents"] = sharing_agents
        
        # Create edge between agent and context
        edge_id = f"{agent_node_id}_{context_node_id}"
        
        if edge_id not in self.edges:
            self.edges[edge_id] = AgentGraphEdge(
                id=edge_id,
                source=agent_node_id,
                target=context_node_id,
                type=AgentEdgeType.CONTEXT_SHARE,
                weight=1.0,
                timestamp=datetime.utcnow(),
                color="#8A2BE2",
                style="dashed",
                metadata={"context_sharing": True}
            )
    
    async def _handle_error_event(self, event: LifecycleEventData) -> None:
        """Handle error and failure events."""
        agent_id = event.agent_id
        if not agent_id:
            return
        
        node_id = f"agent_{agent_id}"
        if node_id in self.nodes:
            self.nodes[node_id].status = AgentNodeStatus.ERROR
            self.nodes[node_id].color = "#FF4444"  # Red for errors
            self.nodes[node_id].metadata.update({
                "error_message": event.payload.get("error"),
                "error_timestamp": event.timestamp
            })
    
    async def _calculate_node_position(
        self, 
        node_id: str, 
        node_type: AgentNodeType
    ) -> Dict[str, float]:
        """Calculate optimal position for a new node."""
        # Simple circular layout based on node type
        existing_nodes_of_type = [
            node for node in self.nodes.values() 
            if node.type == node_type
        ]
        
        count = len(existing_nodes_of_type)
        angle = (count * 2 * 3.14159) / max(8, count + 1)  # Distribute around circle
        
        # Different radii for different node types
        radius_map = {
            AgentNodeType.AGENT: 200,
            AgentNodeType.TOOL: 300,
            AgentNodeType.CONTEXT: 150,
            AgentNodeType.SESSION: 100
        }
        
        radius = radius_map.get(node_type, 200)
        
        return {
            "x": radius * 0.8 * (1 + 0.3 * hash(node_id) % 100 / 100),  # Add some randomness
            "y": radius * 0.8 * (1 + 0.3 * hash(node_id[::-1]) % 100 / 100)
        }
    
    def _calculate_uptime(self, created_at: datetime) -> int:
        """Calculate uptime in seconds."""
        return int((datetime.utcnow() - created_at).total_seconds())
    
    async def _add_communication_event(self, event: AgentCommunicationEvent) -> None:
        """Add communication event to history."""
        self.communication_history.append(event)
        
        # Maintain bounded history
        if len(self.communication_history) > self.max_history:
            self.communication_history = self.communication_history[-self.max_history//2:]
    
    async def _send_initial_graph_state(
        self, 
        websocket: WebSocket, 
        session_id: str,
        event_filter: Optional[EventFilter]
    ) -> None:
        """Send initial graph state to newly connected WebSocket."""
        try:
            # Filter nodes and edges based on session and filter criteria
            filtered_nodes = await self._filter_nodes(session_id, event_filter)
            filtered_edges = await self._filter_edges(session_id, event_filter)
            
            initial_state = {
                "type": "initial_graph_state",
                "data": {
                    "nodes": [node.to_dict() for node in filtered_nodes],
                    "edges": [edge.to_dict() for edge in filtered_edges],
                    "session_colors": self.session_color_manager.session_colors,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            await websocket.send_text(json.dumps(initial_state))
            
        except Exception as e:
            logger.error(
                "Error sending initial graph state",
                session_id=session_id,
                error=str(e)
            )
    
    async def _filter_nodes(
        self, 
        session_id: str, 
        event_filter: Optional[EventFilter]
    ) -> List[AgentGraphNode]:
        """Filter nodes based on session and filter criteria."""
        filtered_nodes = []
        
        for node in self.nodes.values():
            # Session filtering
            node_session = node.metadata.get("session_id", "default")
            if session_id != "all" and node_session != session_id:
                continue
            
            # Apply additional filters if provided
            if event_filter:
                if (event_filter.session_ids and 
                    node_session not in event_filter.session_ids):
                    continue
                
                if (event_filter.node_types and 
                    node.type.value not in event_filter.node_types):
                    continue
            
            filtered_nodes.append(node)
        
        return filtered_nodes
    
    async def _filter_edges(
        self, 
        session_id: str, 
        event_filter: Optional[EventFilter]
    ) -> List[AgentGraphEdge]:
        """Filter edges based on session and filter criteria."""
        filtered_edges = []
        
        for edge in self.edges.values():
            # Only include edges where both nodes are in filtered set
            source_node = self.nodes.get(edge.source)
            target_node = self.nodes.get(edge.target)
            
            if not source_node or not target_node:
                continue
            
            # Session filtering
            source_session = source_node.metadata.get("session_id", "default")
            target_session = target_node.metadata.get("session_id", "default")
            
            if (session_id != "all" and 
                source_session != session_id and 
                target_session != session_id):
                continue
            
            filtered_edges.append(edge)
        
        return filtered_edges
    
    async def _broadcast_graph_update(self, event: LifecycleEventData) -> None:
        """Broadcast graph updates to all connected WebSockets."""
        if not self.active_websockets:
            return
        
        try:
            update_message = {
                "type": "graph_update",
                "data": {
                    "event": {
                        "agent_id": event.agent_id,
                        "session_id": event.session_id,
                        "type": event.event_type.value,
                        "timestamp": event.timestamp,
                        "payload": event.payload
                    },
                    "updated_nodes": {},
                    "updated_edges": {},
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            # Include relevant node and edge updates
            agent_id = event.payload.get("agent_id")
            if agent_id:
                node_id = f"agent_{agent_id}"
                if node_id in self.nodes:
                    update_message["data"]["updated_nodes"][node_id] = self.nodes[node_id].to_dict()
            
            # Send to all connected WebSockets
            disconnected_connections = []
            
            for connection_id, websocket in self.active_websockets.items():
                try:
                    await websocket.send_text(json.dumps(update_message))
                except WebSocketDisconnect:
                    disconnected_connections.append(connection_id)
                except Exception as e:
                    logger.error(
                        "Error sending update to WebSocket",
                        connection_id=connection_id,
                        error=str(e)
                    )
                    disconnected_connections.append(connection_id)
            
            # Clean up disconnected WebSockets
            for connection_id in disconnected_connections:
                await self.unregister_websocket(connection_id)
            
        except Exception as e:
            logger.error(
                "Error broadcasting graph update",
                agent_id=event.agent_id,
                session_id=event.session_id,
                error=str(e)
            )
    
    async def get_session_transcript(
        self, 
        session_id: str,
        agent_filter: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[AgentCommunicationEvent]:
        """Get chronological communication transcript for a session."""
        filtered_events = []
        
        for event in reversed(self.communication_history[-limit*2:]):  # Get more than needed
            if event.session_id != session_id:
                continue
            
            if agent_filter:
                if (event.source_agent_id not in agent_filter and 
                    (event.target_agent_id and event.target_agent_id not in agent_filter)):
                    continue
            
            filtered_events.append(event)
            
            if len(filtered_events) >= limit:
                break
        
        return list(reversed(filtered_events))  # Return in chronological order
    
    async def get_graph_data(
        self, 
        session_id: str = "all",
        event_filter: Optional[EventFilter] = None
    ) -> Dict[str, Any]:
        """Get current graph data with optional filtering."""
        filtered_nodes = await self._filter_nodes(session_id, event_filter)
        filtered_edges = await self._filter_edges(session_id, event_filter)
        
        return {
            "nodes": [node.to_dict() for node in filtered_nodes],
            "edges": [edge.to_dict() for edge in filtered_edges],
            "stats": {
                "total_nodes": len(filtered_nodes),
                "total_edges": len(filtered_edges),
                "active_agents": len([n for n in filtered_nodes if n.type == AgentNodeType.AGENT and n.status == AgentNodeStatus.ACTIVE]),
                "tools_used": len([n for n in filtered_nodes if n.type == AgentNodeType.TOOL]),
                "contexts_shared": len([n for n in filtered_nodes if n.type == AgentNodeType.CONTEXT])
            },
            "session_colors": self.session_color_manager.session_colors,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def cleanup_old_data(self, max_age_hours: int = 24) -> None:
        """Clean up old nodes, edges, and events."""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        # Remove old nodes
        old_nodes = [
            node_id for node_id, node in self.nodes.items()
            if node.last_updated < cutoff_time
        ]
        
        for node_id in old_nodes:
            del self.nodes[node_id]
        
        # Remove old edges
        old_edges = [
            edge_id for edge_id, edge in self.edges.items()
            if edge.timestamp < cutoff_time
        ]
        
        for edge_id in old_edges:
            del self.edges[edge_id]
        
        # Remove old communication events
        self.communication_history = [
            event for event in self.communication_history
            if event.timestamp > cutoff_time
        ]
        
        logger.info(
            "Coordination dashboard cleanup completed",
            removed_nodes=len(old_nodes),
            removed_edges=len(old_edges),
            remaining_events=len(self.communication_history)
        )


# Global coordination dashboard instance
coordination_dashboard = CoordinationDashboard()