"""
Epic B Phase B.4: Real-time WebSocket API for Mobile PWA

Implements WebSocket streaming for live agent monitoring and task updates.
Provides the real-time dashboard functionality required by the Mobile PWA.
Integrates with SimpleOrchestrator for live system status.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Set, Optional
from fastapi import WebSocket, WebSocketDisconnect, APIRouter, Depends
import structlog
import asyncio

from ...core.simple_orchestrator import SimpleOrchestrator, create_simple_orchestrator

logger = structlog.get_logger()
router = APIRouter()

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.agent_subscriptions: Dict[str, Set[str]] = {}  # client_id -> agent_ids
        self.task_subscriptions: Dict[str, Set[str]] = {}   # client_id -> task_ids
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept WebSocket connection and register client."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.agent_subscriptions[client_id] = set()
        self.task_subscriptions[client_id] = set()
        logger.info("WebSocket client connected", client_id=client_id)
    
    def disconnect(self, client_id: str):
        """Remove client from connection manager."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.agent_subscriptions:
            del self.agent_subscriptions[client_id]
        if client_id in self.task_subscriptions:
            del self.task_subscriptions[client_id]
        logger.info("WebSocket client disconnected", client_id=client_id)
    
    async def send_personal_message(self, message: dict, client_id: str):
        """Send message to specific client."""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error("Failed to send WebSocket message", client_id=client_id, error=str(e))
                self.disconnect(client_id)
    
    async def broadcast_agent_update(self, agent_id: str, update_data: dict):
        """Broadcast agent updates to subscribed clients."""
        message = {
            "type": "agent_update",
            "agent_id": agent_id,
            "data": update_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        for client_id, subscribed_agents in self.agent_subscriptions.items():
            if agent_id in subscribed_agents or "*" in subscribed_agents:
                await self.send_personal_message(message, client_id)
    
    async def broadcast_task_update(self, task_id: str, update_data: dict):
        """Broadcast task updates to subscribed clients."""
        message = {
            "type": "task_update", 
            "task_id": task_id,
            "data": update_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        for client_id, subscribed_tasks in self.task_subscriptions.items():
            if task_id in subscribed_tasks or "*" in subscribed_tasks:
                await self.send_personal_message(message, client_id)
    
    async def broadcast_system_status(self, status_data: dict):
        """Broadcast system status to all connected clients."""
        message = {
            "type": "system_status",
            "data": status_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        for client_id in self.active_connections:
            await self.send_personal_message(message, client_id)

# Global connection manager
manager = ConnectionManager()

# Get orchestrator instance
async def get_orchestrator() -> SimpleOrchestrator:
    """Get orchestrator instance."""
    from .agents import get_orchestrator
    return await get_orchestrator()


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    Main WebSocket endpoint for real-time updates.
    
    Epic B Phase B.4: Real-time dashboard communication for Mobile PWA.
    """
    await manager.connect(websocket, client_id)
    
    try:
        # Send welcome message with connection info
        await websocket.send_json({
            "type": "connection_established",
            "client_id": client_id,
            "timestamp": datetime.utcnow().isoformat(),
            "available_commands": [
                "subscribe_agent", "unsubscribe_agent",
                "subscribe_task", "unsubscribe_task", 
                "get_system_status", "list_agents", "list_tasks"
            ]
        })
        
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            await handle_websocket_message(client_id, data)
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info("WebSocket client disconnected", client_id=client_id)
    except Exception as e:
        logger.error("WebSocket error", client_id=client_id, error=str(e))
        manager.disconnect(client_id)


async def handle_websocket_message(client_id: str, message: dict):
    """Handle incoming WebSocket messages from clients."""
    try:
        command = message.get("command")
        if not command:
            await manager.send_personal_message({
                "type": "error",
                "message": "Missing 'command' field"
            }, client_id)
            return
        
        # Handle subscription commands
        if command == "subscribe_agent":
            agent_id = message.get("agent_id", "*")
            manager.agent_subscriptions[client_id].add(agent_id)
            await manager.send_personal_message({
                "type": "subscription_confirmed",
                "subscription_type": "agent",
                "agent_id": agent_id
            }, client_id)
        
        elif command == "unsubscribe_agent":
            agent_id = message.get("agent_id", "*")
            manager.agent_subscriptions[client_id].discard(agent_id)
            await manager.send_personal_message({
                "type": "unsubscription_confirmed",
                "subscription_type": "agent",
                "agent_id": agent_id
            }, client_id)
        
        elif command == "subscribe_task":
            task_id = message.get("task_id", "*")
            manager.task_subscriptions[client_id].add(task_id)
            await manager.send_personal_message({
                "type": "subscription_confirmed",
                "subscription_type": "task",
                "task_id": task_id
            }, client_id)
        
        elif command == "unsubscribe_task":
            task_id = message.get("task_id", "*")
            manager.task_subscriptions[client_id].discard(task_id)
            await manager.send_personal_message({
                "type": "unsubscription_confirmed",
                "subscription_type": "task",
                "task_id": task_id
            }, client_id)
        
        # Handle data request commands
        elif command == "get_system_status":
            orchestrator = await get_orchestrator()
            status = await orchestrator.get_system_status()
            await manager.send_personal_message({
                "type": "system_status",
                "data": status
            }, client_id)
        
        elif command == "list_agents":
            orchestrator = await get_orchestrator()
            agents = []
            for agent_id, agent in orchestrator._agents.items():
                agents.append({
                    "id": agent_id,
                    "role": agent.role.value,
                    "status": agent.status.value,
                    "created_at": agent.created_at.isoformat(),
                    "last_activity": agent.last_activity.isoformat(),
                    "current_task_id": agent.current_task_id
                })
            
            await manager.send_personal_message({
                "type": "agents_list",
                "data": {
                    "agents": agents,
                    "total": len(agents)
                }
            }, client_id)
        
        elif command == "list_tasks":
            orchestrator = await get_orchestrator()
            tasks = []
            for task_id, assignment in orchestrator._task_assignments.items():
                tasks.append({
                    "id": task_id,
                    "status": assignment.status.value,
                    "agent_id": assignment.agent_id if assignment.agent_id != "unassigned" else None,
                    "assigned_at": assignment.assigned_at.isoformat() if assignment.assigned_at else None
                })
            
            await manager.send_personal_message({
                "type": "tasks_list",
                "data": {
                    "tasks": tasks,
                    "total": len(tasks)
                }
            }, client_id)
        
        else:
            await manager.send_personal_message({
                "type": "error", 
                "message": f"Unknown command: {command}"
            }, client_id)
    
    except Exception as e:
        logger.error("Error handling WebSocket message", client_id=client_id, error=str(e))
        await manager.send_personal_message({
            "type": "error",
            "message": "Internal server error"
        }, client_id)


async def start_background_broadcaster():
    """Start background task for periodic system status broadcasts."""
    while True:
        try:
            if manager.active_connections:
                orchestrator = await get_orchestrator()
                status = await orchestrator.get_system_status()
                await manager.broadcast_system_status(status)
            
            # Broadcast every 30 seconds
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error("Background broadcaster error", error=str(e))
            await asyncio.sleep(30)


# Export connection manager for use by other modules
__all__ = ["router", "manager", "ConnectionManager"]