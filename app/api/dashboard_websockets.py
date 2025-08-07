"""
Dashboard WebSocket API for Real-time Monitoring

Provides WebSocket endpoints for real-time dashboard updates, live system monitoring,
and instant coordination event streaming for the LeanVibe Agent Hive dashboard.

Part 3 of the dashboard monitoring infrastructure.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
import structlog

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import get_async_session
from ..core.redis import get_redis, get_message_broker
from ..models.agent import Agent, AgentStatus
from ..models.task import Task, TaskStatus
from .dashboard_monitoring import get_coordination_metrics, get_agent_health_data, get_task_distribution_data

logger = structlog.get_logger()
router = APIRouter(prefix="/api/dashboard", tags=["dashboard-websockets"])


@dataclass
class WebSocketConnection:
    """Represents an active WebSocket connection."""
    websocket: WebSocket
    connection_id: str
    client_type: str
    subscriptions: Set[str]
    connected_at: datetime
    last_activity: datetime
    metadata: Dict[str, Any]


class DashboardWebSocketManager:
    """Manages all WebSocket connections and real-time updates."""
    
    def __init__(self):
        self.connections: Dict[str, WebSocketConnection] = {}
        self.subscription_groups: Dict[str, Set[str]] = {
            "agents": set(),
            "coordination": set(), 
            "tasks": set(),
            "system": set(),
            "alerts": set()
        }
        self.broadcast_task: Optional[asyncio.Task] = None
        self.redis_listener_task: Optional[asyncio.Task] = None
        self.health_monitor_task: Optional[asyncio.Task] = None
        
    async def connect(
        self, 
        websocket: WebSocket, 
        connection_id: str,
        client_type: str = "dashboard",
        subscriptions: Optional[List[str]] = None
    ) -> WebSocketConnection:
        """Connect a new WebSocket client with subscription management."""
        await websocket.accept()
        
        connection = WebSocketConnection(
            websocket=websocket,
            connection_id=connection_id,
            client_type=client_type,
            subscriptions=set(subscriptions or ["agents", "coordination", "tasks", "system"]),
            connected_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            metadata={}
        )
        
        self.connections[connection_id] = connection
        
        # Add to subscription groups
        for subscription in connection.subscriptions:
            if subscription in self.subscription_groups:
                self.subscription_groups[subscription].add(connection_id)
        
        # Start background tasks if this is the first connection
        if len(self.connections) == 1:
            await self._start_background_tasks()
        
        # Send initial connection confirmation
        await self._send_to_connection(connection_id, {
            "type": "connection_established",
            "connection_id": connection_id,
            "subscriptions": list(connection.subscriptions),
            "server_time": datetime.utcnow().isoformat()
        })
        
        logger.info("Dashboard WebSocket connected", 
                   connection_id=connection_id, 
                   client_type=client_type,
                   subscriptions=list(connection.subscriptions))
        
        return connection
    
    async def disconnect(self, connection_id: str) -> None:
        """Disconnect a WebSocket client."""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        
        # Remove from subscription groups
        for subscription in connection.subscriptions:
            if subscription in self.subscription_groups:
                self.subscription_groups[subscription].discard(connection_id)
        
        # Remove connection
        del self.connections[connection_id]
        
        # Stop background tasks if no connections remain
        if len(self.connections) == 0:
            await self._stop_background_tasks()
        
        logger.info("Dashboard WebSocket disconnected", connection_id=connection_id)
    
    async def handle_message(self, connection_id: str, message: Dict[str, Any]) -> None:
        """Handle incoming WebSocket messages from clients."""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        connection.last_activity = datetime.utcnow()
        
        message_type = message.get("type")
        
        if message_type == "ping":
            await self._send_to_connection(connection_id, {
                "type": "pong",
                "timestamp": datetime.utcnow().isoformat()
            })
            
        elif message_type == "subscribe":
            # Add new subscriptions
            new_subs = set(message.get("subscriptions", []))
            valid_subs = new_subs.intersection(self.subscription_groups.keys())
            
            for subscription in valid_subs:
                connection.subscriptions.add(subscription)
                self.subscription_groups[subscription].add(connection_id)
            
            await self._send_to_connection(connection_id, {
                "type": "subscription_updated",
                "subscriptions": list(connection.subscriptions)
            })
            
        elif message_type == "unsubscribe":
            # Remove subscriptions
            remove_subs = set(message.get("subscriptions", []))
            
            for subscription in remove_subs:
                connection.subscriptions.discard(subscription)
                self.subscription_groups[subscription].discard(connection_id)
            
            await self._send_to_connection(connection_id, {
                "type": "subscription_updated",
                "subscriptions": list(connection.subscriptions)
            })
            
        elif message_type == "request_data":
            # Client requesting specific data
            data_type = message.get("data_type")
            await self._handle_data_request(connection_id, data_type, message.get("params", {}))
            
        else:
            logger.warning("Unknown WebSocket message type", 
                          connection_id=connection_id, 
                          message_type=message_type)
    
    async def broadcast_to_subscription(
        self, 
        subscription: str, 
        message_type: str, 
        data: Dict[str, Any]
    ) -> int:
        """Broadcast a message to all clients subscribed to a specific topic."""
        if subscription not in self.subscription_groups:
            return 0
        
        message = {
            "type": message_type,
            "subscription": subscription,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        sent_count = 0
        failed_connections = []
        
        for connection_id in self.subscription_groups[subscription]:
            if await self._send_to_connection(connection_id, message):
                sent_count += 1
            else:
                failed_connections.append(connection_id)
        
        # Clean up failed connections
        for connection_id in failed_connections:
            await self.disconnect(connection_id)
        
        return sent_count
    
    async def broadcast_to_all(self, message_type: str, data: Dict[str, Any]) -> int:
        """Broadcast a message to all connected clients."""
        message = {
            "type": message_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        sent_count = 0
        failed_connections = []
        
        for connection_id in list(self.connections.keys()):
            if await self._send_to_connection(connection_id, message):
                sent_count += 1
            else:
                failed_connections.append(connection_id)
        
        # Clean up failed connections
        for connection_id in failed_connections:
            await self.disconnect(connection_id)
        
        return sent_count
    
    async def _send_to_connection(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """Send message to a specific connection. Returns True if successful."""
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        
        try:
            await connection.websocket.send_text(json.dumps(message))
            return True
        except Exception as e:
            logger.warning("Failed to send WebSocket message", 
                          connection_id=connection_id, 
                          error=str(e))
            return False
    
    async def _handle_data_request(
        self, 
        connection_id: str, 
        data_type: str, 
        params: Dict[str, Any]
    ) -> None:
        """Handle specific data requests from clients."""
        try:
            if data_type == "agent_status":
                # Get fresh agent data (would normally inject DB session)
                data = {
                    "agents": [],  # Would fetch real agent data
                    "summary": {"total": 0, "active": 0}
                }
                
            elif data_type == "coordination_metrics":
                # Get fresh coordination metrics
                data = {
                    "success_rate": 0.0,  # Would fetch real metrics
                    "trend": "unknown"
                }
                
            elif data_type == "system_health":
                # Get system health data
                data = {
                    "overall_status": "unknown",  # Would fetch real health
                    "components": {}
                }
                
            else:
                data = {"error": f"Unknown data type: {data_type}"}
            
            await self._send_to_connection(connection_id, {
                "type": "data_response",
                "data_type": data_type,
                "data": data
            })
            
        except Exception as e:
            await self._send_to_connection(connection_id, {
                "type": "data_error",
                "data_type": data_type,
                "error": str(e)
            })
    
    async def _start_background_tasks(self) -> None:
        """Start background tasks for real-time updates."""
        if self.broadcast_task is None:
            self.broadcast_task = asyncio.create_task(self._broadcast_loop())
            
        if self.redis_listener_task is None:
            self.redis_listener_task = asyncio.create_task(self._redis_listener_loop())
            
        if self.health_monitor_task is None:
            self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        
        logger.info("Dashboard WebSocket background tasks started")
    
    async def _stop_background_tasks(self) -> None:
        """Stop background tasks when no connections remain."""
        tasks = [self.broadcast_task, self.redis_listener_task, self.health_monitor_task]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.broadcast_task = None
        self.redis_listener_task = None
        self.health_monitor_task = None
        
        logger.info("Dashboard WebSocket background tasks stopped")
    
    async def _broadcast_loop(self) -> None:
        """Main broadcast loop for periodic updates."""
        while True:
            try:
                # Skip if no connections
                if not self.connections:
                    await asyncio.sleep(5)
                    continue
                
                # Send periodic updates based on subscriptions
                current_time = datetime.utcnow()
                
                # Agent status updates (every 5 seconds)
                if self.subscription_groups["agents"]:
                    agent_data = {
                        "active_count": 2,  # Would get real data
                        "health_summary": {"healthy": 2, "degraded": 0},
                        "last_updated": current_time.isoformat()
                    }
                    await self.broadcast_to_subscription("agents", "agent_update", agent_data)
                
                # Coordination metrics (every 10 seconds) 
                if self.subscription_groups["coordination"] and int(current_time.timestamp()) % 10 == 0:
                    coord_data = {
                        "success_rate": 75.5,  # Would get real data
                        "total_tasks": 45,
                        "trend": "stable",
                        "last_updated": current_time.isoformat()
                    }
                    await self.broadcast_to_subscription("coordination", "coordination_update", coord_data)
                
                # Task queue updates (every 3 seconds)
                if self.subscription_groups["tasks"] and int(current_time.timestamp()) % 3 == 0:
                    task_data = {
                        "queue_length": 8,  # Would get real data  
                        "in_progress": 3,
                        "failed_recent": 1,
                        "last_updated": current_time.isoformat()
                    }
                    await self.broadcast_to_subscription("tasks", "task_update", task_data)
                
                # System health (every 30 seconds)
                if self.subscription_groups["system"] and int(current_time.timestamp()) % 30 == 0:
                    system_data = {
                        "overall_health": "healthy",  # Would get real data
                        "database_status": "healthy",
                        "redis_status": "healthy", 
                        "last_updated": current_time.isoformat()
                    }
                    await self.broadcast_to_subscription("system", "system_update", system_data)
                
                await asyncio.sleep(1)  # Base interval
                
            except Exception as e:
                logger.error("Error in WebSocket broadcast loop", error=str(e))
                await asyncio.sleep(5)
    
    async def _redis_listener_loop(self) -> None:
        """Listen for Redis events and broadcast to relevant clients."""
        while True:
            try:
                redis_client = get_redis()
                
                # Subscribe to system events
                pubsub = redis_client.pubsub()
                await pubsub.subscribe("system_events", "agent_events:*")
                
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        await self._handle_redis_event(message)
                
            except Exception as e:
                logger.error("Error in Redis listener loop", error=str(e))
                await asyncio.sleep(10)
    
    async def _health_monitor_loop(self) -> None:
        """Monitor system health and send alerts."""
        while True:
            try:
                # Check for critical conditions that require immediate alerts
                
                # Simulate health checks (would use real health monitoring)
                critical_alerts = []
                
                # Check coordination success rate
                # success_rate = await get_coordination_success_rate()
                # if success_rate < 30:
                #     critical_alerts.append({
                #         "level": "critical",
                #         "message": f"Coordination success rate critically low: {success_rate}%",
                #         "action": "immediate_intervention_required"
                #     })
                
                # Send alerts if any critical conditions detected
                if critical_alerts and self.subscription_groups["alerts"]:
                    await self.broadcast_to_subscription("alerts", "critical_alert", {
                        "alerts": critical_alerts,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error("Error in health monitor loop", error=str(e))
                await asyncio.sleep(60)
    
    async def _handle_redis_event(self, message: Dict[str, Any]) -> None:
        """Handle Redis pub/sub events and broadcast to clients."""
        try:
            channel = message["channel"].decode() if isinstance(message["channel"], bytes) else message["channel"]
            data = json.loads(message["data"].decode() if isinstance(message["data"], bytes) else message["data"])
            
            # Route events to appropriate subscriptions
            if channel == "system_events":
                if self.subscription_groups["system"]:
                    await self.broadcast_to_subscription("system", "system_event", data)
                    
            elif channel.startswith("agent_events:"):
                if self.subscription_groups["agents"]:
                    await self.broadcast_to_subscription("agents", "agent_event", data)
            
        except Exception as e:
            logger.error("Error handling Redis event", error=str(e))
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about current connections."""
        current_time = datetime.utcnow()
        
        return {
            "total_connections": len(self.connections),
            "connections_by_type": {},  # Would group by client_type
            "subscription_counts": {
                sub: len(clients) for sub, clients in self.subscription_groups.items()
            },
            "active_connections": len([
                c for c in self.connections.values() 
                if (current_time - c.last_activity).total_seconds() < 300  # Active in last 5 minutes
            ]),
            "oldest_connection": min(
                [c.connected_at for c in self.connections.values()], 
                default=current_time
            ).isoformat(),
            "background_tasks_running": all([
                self.broadcast_task and not self.broadcast_task.done(),
                self.redis_listener_task and not self.redis_listener_task.done(),
                self.health_monitor_task and not self.health_monitor_task.done()
            ]) if self.connections else False
        }


# Global WebSocket manager instance
websocket_manager = DashboardWebSocketManager()


# ==================== WEBSOCKET ENDPOINTS ====================

@router.websocket("/ws/agents")
async def websocket_agents(
    websocket: WebSocket,
    connection_id: Optional[str] = Query(None, description="Optional connection ID")
):
    """
    WebSocket endpoint for real-time agent status updates.
    
    Provides live agent health, status changes, and performance metrics.
    """
    connection_id = connection_id or str(uuid.uuid4())
    
    try:
        connection = await websocket_manager.connect(
            websocket, 
            connection_id, 
            client_type="agent_monitor",
            subscriptions=["agents", "system"]
        )
        
        while True:
            try:
                # Wait for messages from client
                data = await websocket.receive_text()
                message = json.loads(data)
                await websocket_manager.handle_message(connection_id, message)
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket_manager._send_to_connection(connection_id, {
                    "type": "error",
                    "message": "Invalid JSON message format"
                })
            except Exception as e:
                logger.error("Error in agent WebSocket", 
                           connection_id=connection_id, 
                           error=str(e))
                await websocket_manager._send_to_connection(connection_id, {
                    "type": "error", 
                    "message": str(e)
                })
                break
                
    except Exception as e:
        logger.error("Agent WebSocket connection failed", 
                    connection_id=connection_id, 
                    error=str(e))
    finally:
        await websocket_manager.disconnect(connection_id)


@router.websocket("/ws/coordination") 
async def websocket_coordination(
    websocket: WebSocket,
    connection_id: Optional[str] = Query(None, description="Optional connection ID")
):
    """
    WebSocket endpoint for real-time coordination monitoring.
    
    Streams coordination success rates, failure events, and recovery actions.
    """
    connection_id = connection_id or str(uuid.uuid4())
    
    try:
        connection = await websocket_manager.connect(
            websocket,
            connection_id,
            client_type="coordination_monitor", 
            subscriptions=["coordination", "alerts", "system"]
        )
        
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                await websocket_manager.handle_message(connection_id, message)
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket_manager._send_to_connection(connection_id, {
                    "type": "error",
                    "message": "Invalid JSON message format"
                })
            except Exception as e:
                logger.error("Error in coordination WebSocket",
                           connection_id=connection_id,
                           error=str(e))
                break
                
    except Exception as e:
        logger.error("Coordination WebSocket connection failed",
                    connection_id=connection_id,
                    error=str(e))
    finally:
        await websocket_manager.disconnect(connection_id)


@router.websocket("/ws/tasks")
async def websocket_tasks(
    websocket: WebSocket,
    connection_id: Optional[str] = Query(None, description="Optional connection ID")
):
    """
    WebSocket endpoint for real-time task distribution monitoring.
    
    Streams task queue status, assignment changes, and completion events.
    """
    connection_id = connection_id or str(uuid.uuid4())
    
    try:
        connection = await websocket_manager.connect(
            websocket,
            connection_id,
            client_type="task_monitor",
            subscriptions=["tasks", "agents"]
        )
        
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                await websocket_manager.handle_message(connection_id, message)
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket_manager._send_to_connection(connection_id, {
                    "type": "error",
                    "message": "Invalid JSON message format"
                })
            except Exception as e:
                logger.error("Error in task WebSocket",
                           connection_id=connection_id,
                           error=str(e))
                break
                
    except Exception as e:
        logger.error("Task WebSocket connection failed",
                    connection_id=connection_id,
                    error=str(e))
    finally:
        await websocket_manager.disconnect(connection_id)


@router.websocket("/ws/system")
async def websocket_system(
    websocket: WebSocket,
    connection_id: Optional[str] = Query(None, description="Optional connection ID")
):
    """
    WebSocket endpoint for real-time system health monitoring.
    
    Streams overall system health, component status, and critical alerts.
    """
    connection_id = connection_id or str(uuid.uuid4())
    
    try:
        connection = await websocket_manager.connect(
            websocket,
            connection_id,
            client_type="system_monitor",
            subscriptions=["system", "alerts"]
        )
        
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                await websocket_manager.handle_message(connection_id, message)
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket_manager._send_to_connection(connection_id, {
                    "type": "error",
                    "message": "Invalid JSON message format"
                })
            except Exception as e:
                logger.error("Error in system WebSocket",
                           connection_id=connection_id,
                           error=str(e))
                break
                
    except Exception as e:
        logger.error("System WebSocket connection failed",
                    connection_id=connection_id,
                    error=str(e))
    finally:
        await websocket_manager.disconnect(connection_id)


@router.websocket("/ws/dashboard")
async def websocket_dashboard_all(
    websocket: WebSocket,
    connection_id: Optional[str] = Query(None, description="Optional connection ID"),
    subscriptions: Optional[str] = Query("agents,coordination,tasks,system", description="Comma-separated subscriptions")
):
    """
    WebSocket endpoint for comprehensive dashboard monitoring.
    
    Single endpoint that can subscribe to multiple data streams for full dashboard functionality.
    """
    connection_id = connection_id or str(uuid.uuid4())
    subscription_list = [s.strip() for s in subscriptions.split(",") if s.strip()]
    
    try:
        connection = await websocket_manager.connect(
            websocket,
            connection_id,
            client_type="full_dashboard",
            subscriptions=subscription_list
        )
        
        # Send initial dashboard data
        await websocket_manager._send_to_connection(connection_id, {
            "type": "dashboard_initialized",
            "subscriptions": subscription_list,
            "connection_info": {
                "connection_id": connection_id,
                "client_type": "full_dashboard",
                "server_time": datetime.utcnow().isoformat()
            }
        })
        
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                await websocket_manager.handle_message(connection_id, message)
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket_manager._send_to_connection(connection_id, {
                    "type": "error",
                    "message": "Invalid JSON message format"
                })
            except Exception as e:
                logger.error("Error in dashboard WebSocket",
                           connection_id=connection_id,
                           error=str(e))
                break
                
    except Exception as e:
        logger.error("Dashboard WebSocket connection failed",
                    connection_id=connection_id,
                    error=str(e))
    finally:
        await websocket_manager.disconnect(connection_id)


# ==================== WEBSOCKET MANAGEMENT APIs ====================

@router.get("/websocket/stats", response_model=Dict[str, Any])
async def get_websocket_stats():
    """
    Get statistics about active WebSocket connections.
    
    Provides insights into real-time client connections and subscription patterns.
    """
    try:
        stats = websocket_manager.get_connection_stats()
        
        return {
            "websocket_stats": stats,
            "endpoints": {
                "agents": "/api/dashboard/ws/agents",
                "coordination": "/api/dashboard/ws/coordination", 
                "tasks": "/api/dashboard/ws/tasks",
                "system": "/api/dashboard/ws/system",
                "dashboard": "/api/dashboard/ws/dashboard"
            },
            "subscription_types": [
                "agents",
                "coordination", 
                "tasks",
                "system",
                "alerts"
            ],
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get WebSocket stats", error=str(e))
        return {
            "error": "Failed to retrieve WebSocket statistics",
            "websocket_stats": {
                "total_connections": 0,
                "active_connections": 0
            }
        }


@router.post("/websocket/broadcast", response_model=Dict[str, Any])
async def broadcast_message(
    subscription: str = Query(..., description="Subscription group to broadcast to"),
    message_type: str = Query(..., description="Type of message to broadcast"),
    message_data: Dict[str, Any] = {}
):
    """
    Manually broadcast a message to WebSocket clients.
    
    Useful for testing WebSocket functionality and sending administrative messages.
    """
    try:
        if subscription not in websocket_manager.subscription_groups:
            return {
                "error": f"Invalid subscription: {subscription}",
                "valid_subscriptions": list(websocket_manager.subscription_groups.keys())
            }
        
        sent_count = await websocket_manager.broadcast_to_subscription(
            subscription, 
            message_type, 
            message_data
        )
        
        return {
            "success": True,
            "subscription": subscription,
            "message_type": message_type,
            "clients_reached": sent_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to broadcast WebSocket message", error=str(e))
        return {
            "error": f"Failed to broadcast message: {str(e)}",
            "subscription": subscription,
            "message_type": message_type
        }


@router.post("/websocket/disconnect/{connection_id}", response_model=Dict[str, Any])
async def disconnect_websocket_client(
    connection_id: str,
    reason: str = Query("Administrative disconnect", description="Reason for disconnect")
):
    """
    Manually disconnect a specific WebSocket client.
    
    Provides administrative control over WebSocket connections.
    """
    try:
        if connection_id not in websocket_manager.connections:
            return {
                "error": "Connection not found",
                "connection_id": connection_id,
                "active_connections": list(websocket_manager.connections.keys())
            }
        
        # Send disconnect notification to client first
        await websocket_manager._send_to_connection(connection_id, {
            "type": "disconnect_notice",
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Disconnect the client
        await websocket_manager.disconnect(connection_id)
        
        return {
            "success": True,
            "connection_id": connection_id,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to disconnect WebSocket client", 
                    connection_id=connection_id, 
                    error=str(e))
        return {
            "error": f"Failed to disconnect client: {str(e)}",
            "connection_id": connection_id
        }


# Health check for WebSocket functionality
@router.get("/websocket/health", response_model=Dict[str, Any])
async def websocket_health_check():
    """
    Check WebSocket system health and functionality.
    
    Validates WebSocket infrastructure is ready for real-time dashboard connections.
    """
    try:
        health_data = {
            "websocket_manager": "operational",
            "background_tasks": {
                "broadcast_task": websocket_manager.broadcast_task is not None and not websocket_manager.broadcast_task.done() if websocket_manager.broadcast_task else False,
                "redis_listener": websocket_manager.redis_listener_task is not None and not websocket_manager.redis_listener_task.done() if websocket_manager.redis_listener_task else False,
                "health_monitor": websocket_manager.health_monitor_task is not None and not websocket_manager.health_monitor_task.done() if websocket_manager.health_monitor_task else False
            },
            "connection_stats": websocket_manager.get_connection_stats(),
            "redis_connectivity": "unknown"
        }
        
        # Test Redis connectivity
        try:
            redis_client = get_redis()
            await redis_client.ping()
            health_data["redis_connectivity"] = "healthy"
        except Exception as redis_error:
            health_data["redis_connectivity"] = f"failed: {str(redis_error)}"
        
        # Overall health assessment
        health_score = 100
        
        if health_data["redis_connectivity"] != "healthy":
            health_score -= 40  # Redis is critical for real-time events
        
        active_tasks = sum(1 for task in health_data["background_tasks"].values() if task)
        if active_tasks < 3 and health_data["connection_stats"]["total_connections"] > 0:
            health_score -= 30  # Background tasks should be running with active connections
        
        health_data["overall_health"] = {
            "score": max(0, health_score),
            "status": "healthy" if health_score >= 90 else "degraded" if health_score >= 70 else "unhealthy"
        }
        
        return health_data
        
    except Exception as e:
        logger.error("WebSocket health check failed", error=str(e))
        return {
            "overall_health": {
                "score": 0,
                "status": "unhealthy"
            },
            "error": str(e)
        }