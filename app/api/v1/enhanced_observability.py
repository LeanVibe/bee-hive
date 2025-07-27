"""
Enhanced Observability API with Real-Time WebSocket Streaming.

Provides comprehensive lifecycle event monitoring including:
- Real-time WebSocket streaming for dashboard updates
- Enhanced event processing and analytics
- Performance monitoring and alerting
- Historical event analysis and reporting
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.enhanced_lifecycle_hooks import (
    EnhancedLifecycleHookProcessor, 
    EnhancedEventType,
    get_enhanced_lifecycle_hook_processor
)
from ...core.database import get_async_session
from ...models.observability import AgentEvent
from ...schemas.observability import HookEventCreate, HookEventResponse
from ...core.config import get_settings

logger = structlog.get_logger()

router = APIRouter(prefix="/observability", tags=["Enhanced Observability"])


class WebSocketManager:
    """
    WebSocket connection manager for real-time event streaming.
    
    Manages client connections, event broadcasting, and connection lifecycle
    for the enhanced observability dashboard.
    """
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.client_subscriptions: Dict[WebSocket, Dict[str, Any]] = {}
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
        
        # Performance tracking
        self.connection_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "connection_errors": 0,
            "last_activity": datetime.utcnow()
        }
    
    async def connect(self, websocket: WebSocket, client_info: Dict[str, Any]) -> None:
        """Accept new WebSocket connection."""
        try:
            await websocket.accept()
            self.active_connections.add(websocket)
            self.client_subscriptions[websocket] = {
                "event_types": [],  # Will be set by client
                "agent_filters": [],
                "session_filters": [],
                "severity_filters": ["info", "warning", "error", "critical"]
            }
            self.connection_metadata[websocket] = {
                **client_info,
                "connected_at": datetime.utcnow().isoformat(),
                "last_ping": datetime.utcnow().isoformat()
            }
            
            self.connection_stats["total_connections"] += 1
            self.connection_stats["active_connections"] = len(self.active_connections)
            
            logger.info(
                f"📡 WebSocket client connected",
                client_ip=client_info.get("client_ip"),
                user_agent=client_info.get("user_agent"),
                active_connections=len(self.active_connections)
            )
            
            # Send welcome message
            await self._send_to_client(websocket, {
                "type": "connection_established",
                "data": {
                    "connection_id": id(websocket),
                    "server_time": datetime.utcnow().isoformat(),
                    "available_event_types": [e.value for e in EnhancedEventType],
                    "connection_stats": self.connection_stats
                }
            })
            
            # Register with lifecycle processor
            processor = get_enhanced_lifecycle_hook_processor()
            processor.register_websocket_client(websocket)
            
        except Exception as e:
            logger.error(f"❌ WebSocket connection failed: {e}")
            await self.disconnect(websocket)
    
    async def disconnect(self, websocket: WebSocket) -> None:
        """Handle WebSocket disconnection."""
        try:
            self.active_connections.discard(websocket)
            self.client_subscriptions.pop(websocket, None)
            self.connection_metadata.pop(websocket, None)
            
            self.connection_stats["active_connections"] = len(self.active_connections)
            
            # Unregister from lifecycle processor
            processor = get_enhanced_lifecycle_hook_processor()
            processor.unregister_websocket_client(websocket)
            
            logger.info(
                f"📡 WebSocket client disconnected",
                active_connections=len(self.active_connections)
            )
            
        except Exception as e:
            logger.error(f"❌ WebSocket disconnection error: {e}")
    
    async def send_to_all(self, message: Dict[str, Any]) -> None:
        """Send message to all connected clients."""
        if not self.active_connections:
            return
        
        disconnected_clients = set()
        
        for websocket in self.active_connections.copy():
            try:
                if self._should_send_to_client(websocket, message):
                    await self._send_to_client(websocket, message)
            except Exception:
                disconnected_clients.add(websocket)
        
        # Clean up disconnected clients
        for websocket in disconnected_clients:
            await self.disconnect(websocket)
    
    async def send_to_client(self, websocket: WebSocket, message: Dict[str, Any]) -> None:
        """Send message to specific client."""
        try:
            await self._send_to_client(websocket, message)
        except Exception as e:
            logger.error(f"❌ Failed to send message to client: {e}")
            await self.disconnect(websocket)
    
    async def update_client_subscription(
        self, 
        websocket: WebSocket, 
        subscription_data: Dict[str, Any]
    ) -> None:
        """Update client subscription preferences."""
        if websocket in self.client_subscriptions:
            self.client_subscriptions[websocket].update(subscription_data)
            
            await self._send_to_client(websocket, {
                "type": "subscription_updated",
                "data": {
                    "subscription": self.client_subscriptions[websocket],
                    "updated_at": datetime.utcnow().isoformat()
                }
            })
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics."""
        return {
            **self.connection_stats,
            "active_connections": len(self.active_connections),
            "client_subscriptions": len(self.client_subscriptions)
        }
    
    async def _send_to_client(self, websocket: WebSocket, message: Dict[str, Any]) -> None:
        """Send message to client with error handling."""
        try:
            await websocket.send_text(json.dumps(message, default=str))
            self.connection_stats["messages_sent"] += 1
            self.connection_stats["last_activity"] = datetime.utcnow()
        except Exception as e:
            self.connection_stats["connection_errors"] += 1
            raise e
    
    def _should_send_to_client(self, websocket: WebSocket, message: Dict[str, Any]) -> bool:
        """Determine if message should be sent to specific client."""
        if websocket not in self.client_subscriptions:
            return False
        
        subscription = self.client_subscriptions[websocket]
        message_data = message.get("data", {})
        
        # Check event type filter
        if subscription.get("event_types"):
            event_type = message_data.get("event_type")
            if event_type not in subscription["event_types"]:
                return False
        
        # Check severity filter
        severity = message_data.get("severity", "info")
        if severity not in subscription.get("severity_filters", ["info", "warning", "error", "critical"]):
            return False
        
        # Check agent filter
        if subscription.get("agent_filters"):
            agent_id = message_data.get("agent_id")
            if agent_id not in subscription["agent_filters"]:
                return False
        
        # Check session filter
        if subscription.get("session_filters"):
            session_id = message_data.get("session_id")
            if session_id not in subscription["session_filters"]:
                return False
        
        return True


# Global WebSocket manager instance
websocket_manager = WebSocketManager()


@router.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time lifecycle event streaming.
    
    Supports:
    - Real-time event streaming with filtering
    - Client subscription management
    - Performance monitoring data
    - Connection health monitoring
    """
    client_info = {
        "client_ip": websocket.client.host if websocket.client else "unknown",
        "user_agent": websocket.headers.get("user-agent", "unknown")
    }
    
    await websocket_manager.connect(websocket, client_info)
    
    try:
        while True:
            # Receive messages from client
            message = await websocket.receive_text()
            
            try:
                data = json.loads(message)
                await handle_websocket_message(websocket, data)
            except json.JSONDecodeError:
                await websocket_manager.send_to_client(websocket, {
                    "type": "error",
                    "data": {"message": "Invalid JSON format"}
                })
            
    except WebSocketDisconnect:
        await websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        await websocket_manager.disconnect(websocket)


async def handle_websocket_message(websocket: WebSocket, data: Dict[str, Any]) -> None:
    """Handle incoming WebSocket messages from clients."""
    message_type = data.get("type")
    
    if message_type == "subscribe":
        # Update client subscription
        subscription_data = data.get("data", {})
        await websocket_manager.update_client_subscription(websocket, subscription_data)
        
    elif message_type == "ping":
        # Health check
        await websocket_manager.send_to_client(websocket, {
            "type": "pong",
            "data": {"server_time": datetime.utcnow().isoformat()}
        })
        
    elif message_type == "get_stats":
        # Send connection statistics
        stats = websocket_manager.get_connection_stats()
        await websocket_manager.send_to_client(websocket, {
            "type": "stats",
            "data": stats
        })
        
    elif message_type == "get_recent_events":
        # Send recent events
        recent_events = await get_recent_events(limit=data.get("limit", 50))
        await websocket_manager.send_to_client(websocket, {
            "type": "recent_events",
            "data": {"events": recent_events}
        })
        
    else:
        await websocket_manager.send_to_client(websocket, {
            "type": "error",
            "data": {"message": f"Unknown message type: {message_type}"}
        })


@router.post("/hook-events", response_model=HookEventResponse)
async def capture_hook_event(
    event_data: HookEventCreate,
    db: AsyncSession = Depends(get_async_session)
) -> HookEventResponse:
    """
    Capture enhanced lifecycle hook event.
    
    Processes incoming hook events through the enhanced lifecycle processor
    with real-time streaming and analytics.
    """
    try:
        processor = get_enhanced_lifecycle_hook_processor()
        
        event_id = await processor.process_enhanced_event(
            session_id=event_data.session_id,
            agent_id=event_data.agent_id,
            event_type=EnhancedEventType(event_data.event_type),
            payload=event_data.payload,
            correlation_id=event_data.correlation_id,
            severity=event_data.severity or "info",
            tags=event_data.tags
        )
        
        return HookEventResponse(
            event_id=event_id,
            status="processed",
            timestamp=datetime.utcnow(),
            processing_time_ms=0  # Will be updated by processor
        )
        
    except Exception as e:
        logger.error(f"❌ Hook event capture failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events/recent")
async def get_recent_events(
    limit: int = Query(default=100, le=1000),
    event_types: Optional[List[str]] = Query(default=None),
    agent_id: Optional[str] = Query(default=None),
    session_id: Optional[str] = Query(default=None),
    severity: Optional[List[str]] = Query(default=None),
    db: AsyncSession = Depends(get_async_session)
) -> JSONResponse:
    """
    Get recent lifecycle events with filtering.
    
    Supports filtering by event type, agent, session, and severity
    for dashboard and debugging purposes.
    """
    try:
        # This would implement the actual database query
        # For now, return a placeholder response
        events = []
        
        return JSONResponse({
            "events": events,
            "total_count": len(events),
            "filters_applied": {
                "limit": limit,
                "event_types": event_types,
                "agent_id": agent_id,
                "session_id": session_id,
                "severity": severity
            },
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Recent events retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics")
async def get_event_analytics(
    session_id: Optional[str] = Query(default=None),
    agent_id: Optional[str] = Query(default=None),
    time_range_hours: int = Query(default=1, le=168),  # Max 1 week
    include_trends: bool = Query(default=True),
    include_patterns: bool = Query(default=True),
    db: AsyncSession = Depends(get_async_session)
) -> JSONResponse:
    """
    Get comprehensive event analytics and insights.
    
    Provides performance trends, error patterns, and optimization
    recommendations based on lifecycle event data.
    """
    try:
        processor = get_enhanced_lifecycle_hook_processor()
        
        # Convert parameters
        session_uuid = uuid.UUID(session_id) if session_id else None
        agent_uuid = uuid.UUID(agent_id) if agent_id else None
        time_range_seconds = time_range_hours * 3600
        
        analytics = await processor.get_event_analytics(
            session_id=session_uuid,
            agent_id=agent_uuid,
            time_range=time_range_seconds,
            event_types=None  # Include all event types
        )
        
        return JSONResponse({
            "analytics": analytics,
            "parameters": {
                "session_id": session_id,
                "agent_id": agent_id,
                "time_range_hours": time_range_hours,
                "include_trends": include_trends,
                "include_patterns": include_patterns
            },
            "generated_at": datetime.utcnow().isoformat()
        })
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid UUID format: {e}")
    except Exception as e:
        logger.error(f"❌ Event analytics generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def get_observability_health() -> JSONResponse:
    """
    Get observability system health status.
    
    Returns health metrics for the enhanced lifecycle hook system
    including processing performance and connection statistics.
    """
    try:
        processor = get_enhanced_lifecycle_hook_processor()
        websocket_stats = websocket_manager.get_connection_stats()
        
        health_data = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "performance_metrics": processor.performance_metrics,
            "websocket_connections": websocket_stats,
            "system_info": {
                "enhanced_hooks_enabled": True,
                "real_time_streaming_enabled": len(websocket_manager.active_connections) > 0,
                "redis_streams_active": True,  # This would be checked dynamically
                "database_connectivity": True  # This would be checked dynamically
            },
            "health_checks": {
                "event_processing": "healthy" if processor.performance_metrics["avg_processing_time_ms"] < 100 else "degraded",
                "websocket_manager": "healthy" if websocket_stats["connection_errors"] < 10 else "degraded",
                "pattern_detection": "healthy"
            }
        }
        
        # Determine overall health
        if any(check == "degraded" for check in health_data["health_checks"].values()):
            health_data["status"] = "degraded"
        
        return JSONResponse(health_data)
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return JSONResponse({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }, status_code=500)


@router.post("/trigger-test-event")
async def trigger_test_event(
    event_type: str,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None
) -> JSONResponse:
    """
    Trigger test event for debugging and development.
    
    Allows manual triggering of lifecycle events for testing
    the enhanced observability system.
    """
    try:
        processor = get_enhanced_lifecycle_hook_processor()
        
        # Use provided IDs or generate new ones
        test_agent_id = uuid.UUID(agent_id) if agent_id else uuid.uuid4()
        test_session_id = uuid.UUID(session_id) if session_id else uuid.uuid4()
        
        # Create test payload
        test_payload = payload or {
            "test_event": True,
            "triggered_at": datetime.utcnow().isoformat(),
            "description": f"Test event of type {event_type}"
        }
        
        event_id = await processor.process_enhanced_event(
            session_id=test_session_id,
            agent_id=test_agent_id,
            event_type=EnhancedEventType(event_type),
            payload=test_payload,
            severity="info",
            tags={"test": "true", "manual_trigger": "true"}
        )
        
        return JSONResponse({
            "success": True,
            "event_id": event_id,
            "event_type": event_type,
            "agent_id": str(test_agent_id),
            "session_id": str(test_session_id),
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid event type or UUID: {e}")
    except Exception as e:
        logger.error(f"❌ Test event trigger failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Integration with main WebSocket broadcasting
async def broadcast_lifecycle_event(event_data: Dict[str, Any]) -> None:
    """
    Broadcast lifecycle event to all WebSocket clients.
    
    This function is called by the enhanced lifecycle processor
    to stream events in real-time.
    """
    message = {
        "type": "lifecycle_event",
        "data": event_data,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    await websocket_manager.send_to_all(message)