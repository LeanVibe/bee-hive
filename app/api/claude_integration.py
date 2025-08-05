"""
Claude Code Integration API for LeanVibe Agent Hive 2.0

This API provides endpoints specifically for Claude Code hooks integration,
enabling seamless coordination between Claude Code and the autonomous development platform.
"""

from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
import structlog
import json

from ..core.redis import get_redis
from ..core.agent_spawner import get_active_agents_status
from ..services.event_collector_service import EventCollectorService

logger = structlog.get_logger()
router = APIRouter()


class ClaudeSessionRequest(BaseModel):
    """Request to start or resume a Claude Code session."""
    session_id: Optional[str] = Field(default=None, description="Existing session ID for resume")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Session context data")
    hooks_enabled: bool = Field(default=True, description="Whether hooks are enabled for this session")
    mobile_integration: bool = Field(default=True, description="Enable mobile dashboard integration")


class ClaudeSessionResponse(BaseModel):
    """Response from Claude session management."""
    success: bool
    session_id: str
    message: str
    mobile_dashboard_url: Optional[str] = None
    active_agents: Optional[int] = None
    integration_status: Dict[str, Any]


class MobileNotificationRequest(BaseModel):
    """Request to send mobile notification."""
    priority: str = Field(..., description="Notification priority: critical, high, medium")
    title: str = Field(..., description="Notification title")
    message: str = Field(..., description="Notification message")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Additional notification data")
    hook_event: Optional[str] = Field(default=None, description="Originating hook event")


class AgentUpdateRequest(BaseModel):
    """Request to update mobile dashboard with agent status."""
    agent_id: str = Field(..., description="Agent identifier")
    status: str = Field(..., description="Agent status: active, stopped, error")
    event: str = Field(..., description="Event type: start, stop, error")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Additional event data")


@router.post("/claude/session-start", response_model=ClaudeSessionResponse)
async def start_claude_session(request: ClaudeSessionRequest):
    """
    Start a new Claude Code session with LeanVibe Agent Hive integration.
    
    This endpoint initializes the connection between Claude Code and the autonomous
    development platform, enabling hook coordination and mobile oversight.
    """
    try:
        # Generate session ID if not provided
        session_id = request.session_id or f"claude-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        
        logger.info("🚀 Starting Claude Code session", session_id=session_id)
        
        # Get current system status
        redis_client = get_redis()
        active_agents = await get_active_agents_status()
        agent_count = len(active_agents) if active_agents else 0
        
        # Store session information in Redis
        session_data = {
            "session_id": session_id,
            "started_at": datetime.utcnow().isoformat(),
            "hooks_enabled": request.hooks_enabled,
            "mobile_integration": request.mobile_integration,
            "context": request.context or {},
            "active_agents": agent_count
        }
        
        await redis_client.setex(
            f"claude:session:{session_id}",
            3600,  # 1 hour expiry
            json.dumps(session_data)
        )
        
        # Broadcast session start event for mobile dashboard
        if request.mobile_integration:
            event_data = {
                "type": "claude_session_start",
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "agent_count": agent_count,
                "hooks_enabled": request.hooks_enabled
            }
            
            await redis_client.publish("system_events", json.dumps(event_data))
        
        # Prepare mobile dashboard URL
        mobile_url = f"http://localhost:8000/dashboard/mobile?session={session_id}" if request.mobile_integration else None
        
        integration_status = {
            "redis_connected": True,
            "agents_available": agent_count > 0,
            "mobile_dashboard": request.mobile_integration,
            "hooks_system": request.hooks_enabled,
            "event_streaming": True
        }
        
        logger.info("✅ Claude Code session started successfully", 
                   session_id=session_id, 
                   agent_count=agent_count,
                   mobile_integration=request.mobile_integration)
        
        return ClaudeSessionResponse(
            success=True,
            session_id=session_id,
            message=f"LeanVibe Agent Hive 2.0 active - Enhanced mobile oversight available ({agent_count} agents)",
            mobile_dashboard_url=mobile_url,
            active_agents=agent_count,
            integration_status=integration_status
        )
        
    except Exception as e:
        logger.error("❌ Failed to start Claude session", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start Claude session: {str(e)}"
        )


@router.post("/claude/session-resume", response_model=ClaudeSessionResponse)
async def resume_claude_session(request: ClaudeSessionRequest):
    """
    Resume an existing Claude Code session with state restoration.
    
    Restores agent states and mobile dashboard connectivity for continued
    autonomous development coordination.
    """
    try:
        if not request.session_id:
            raise HTTPException(status_code=400, detail="Session ID required for resume")
        
        logger.info("🔄 Resuming Claude Code session", session_id=request.session_id)
        
        # Retrieve session data from Redis
        redis_client = get_redis()
        session_key = f"claude:session:{request.session_id}"
        session_data_str = await redis_client.get(session_key)
        
        if not session_data_str:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        
        session_data = json.loads(session_data_str)
        
        # Update session with resume timestamp
        session_data["resumed_at"] = datetime.utcnow().isoformat()
        session_data["resume_count"] = session_data.get("resume_count", 0) + 1
        
        # Get current system status
        active_agents = await get_active_agents_status()
        agent_count = len(active_agents) if active_agents else 0
        session_data["active_agents"] = agent_count
        
        # Update session in Redis
        await redis_client.setex(session_key, 3600, json.dumps(session_data))
        
        # Broadcast session resume event
        if session_data.get("mobile_integration", True):
            event_data = {
                "type": "claude_session_resume",
                "session_id": request.session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "agent_count": agent_count,
                "resume_count": session_data["resume_count"]
            }
            
            await redis_client.publish("system_events", json.dumps(event_data))
        
        mobile_url = f"http://localhost:8000/dashboard/mobile?session={request.session_id}"
        
        integration_status = {
            "session_restored": True,
            "redis_connected": True,
            "agents_available": agent_count > 0,
            "mobile_dashboard": session_data.get("mobile_integration", True),
            "hooks_system": session_data.get("hooks_enabled", True),
            "resume_count": session_data["resume_count"]
        }
        
        logger.info("✅ Claude Code session resumed successfully", 
                   session_id=request.session_id,
                   agent_count=agent_count,
                   resume_count=session_data["resume_count"])
        
        return ClaudeSessionResponse(
            success=True,
            session_id=request.session_id,
            message=f"Session restored - {agent_count} agents active, resume #{session_data['resume_count']}",
            mobile_dashboard_url=mobile_url,
            active_agents=agent_count,
            integration_status=integration_status
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ Failed to resume Claude session", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to resume Claude session: {str(e)}"
        )


@router.post("/mobile/notifications")
async def send_mobile_notification(request: MobileNotificationRequest):
    """
    Send push notification to mobile dashboard.
    
    Used by Claude Code hooks to send real-time notifications to mobile
    devices for critical events and system alerts.
    """
    try:
        logger.info("📱 Sending mobile notification", 
                   priority=request.priority,
                   title=request.title)
        
        # Prepare notification data
        notification_data = {
            "id": f"notification-{datetime.utcnow().timestamp()}",
            "type": request.data.get("type", "system_notification") if request.data else "system_notification",
            "priority": request.priority,
            "title": request.title,
            "message": request.message,
            "context": request.data or {},
            "timestamp": datetime.utcnow().isoformat(),
            "hook_event": request.hook_event,
            "actions": []
        }
        
        # Add context-specific actions based on priority and event
        if request.priority == "critical":
            notification_data["actions"] = [
                {
                    "id": "investigate",
                    "label": "Investigate",
                    "command": "/hive:investigate",
                    "priority": "primary"
                },
                {
                    "id": "escalate",
                    "label": "Escalate",
                    "command": "/hive:escalate",
                    "priority": "danger"
                }
            ]
        elif request.hook_event in ["PreToolUse", "PostToolUse"]:
            notification_data["actions"] = [
                {
                    "id": "review",
                    "label": "Review",
                    "command": "/hive:review",
                    "priority": "primary"
                }
            ]
        
        # Send via Redis pub/sub for real-time delivery
        redis_client = get_redis()
        
        # Send to mobile notification channel
        await redis_client.publish("mobile_notifications", json.dumps(notification_data))
        
        # Send to general system events for dashboard updates
        await redis_client.publish("system_events", json.dumps({
            "type": "mobile_notification",
            "notification": notification_data
        }))
        
        # Store notification for retrieval via API
        await redis_client.setex(
            f"notification:{notification_data['id']}",
            300,  # 5 minutes
            json.dumps(notification_data)
        )
        
        logger.info("✅ Mobile notification sent successfully", 
                   notification_id=notification_data["id"],
                   priority=request.priority)
        
        return {
            "success": True,
            "notification_id": notification_data["id"],
            "message": "Mobile notification sent successfully",
            "delivery_channels": ["mobile_notifications", "system_events"],
            "actions_available": len(notification_data["actions"])
        }
        
    except Exception as e:
        logger.error("❌ Failed to send mobile notification", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send mobile notification: {str(e)}"
        )


@router.post("/mobile/agent-update")
async def update_mobile_agent_status(request: AgentUpdateRequest):
    """
    Update mobile dashboard with agent status changes.
    
    Used by Claude Code hooks to provide real-time agent coordination
    updates to mobile oversight dashboard.
    """
    try:
        logger.info("🤖 Updating mobile agent status", 
                   agent_id=request.agent_id,
                   status=request.status,
                   event=request.event)
        
        # Prepare agent update data
        update_data = {
            "type": "agent_update",
            "agent_id": request.agent_id,
            "status": request.status,
            "event": request.event,
            "timestamp": datetime.utcnow().isoformat(),
            "data": request.data or {}
        }
        
        # Get current agent status for context
        try:
            active_agents = await get_active_agents_status()
            update_data["total_agents"] = len(active_agents) if active_agents else 0
            
            # Find specific agent info if available
            if active_agents:
                agent_info = next((agent for agent in active_agents if agent.get("id") == request.agent_id), None)
                if agent_info:
                    update_data["agent_info"] = agent_info
        except Exception as agent_error:
            logger.warning("Could not retrieve agent status", error=str(agent_error))
        
        # Send real-time update via Redis
        redis_client = get_redis()
        
        # Send to mobile dashboard channel
        await redis_client.publish("mobile_agent_updates", json.dumps(update_data))
        
        # Send to general system events
        await redis_client.publish("system_events", json.dumps(update_data))
        
        # Store recent agent update for dashboard
        await redis_client.setex(
            f"agent_update:{request.agent_id}",
            3600,  # 1 hour
            json.dumps(update_data)
        )
        
        logger.info("✅ Mobile agent update sent successfully", 
                   agent_id=request.agent_id,
                   event=request.event)
        
        return {
            "success": True,
            "agent_id": request.agent_id,
            "message": f"Agent {request.event} update sent to mobile dashboard",
            "timestamp": update_data["timestamp"],
            "total_agents": update_data.get("total_agents", 0)
        }
        
    except Exception as e:
        logger.error("❌ Failed to update mobile agent status", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update mobile agent status: {str(e)}"
        )


@router.get("/mobile/status")
async def get_mobile_dashboard_status():
    """
    Get mobile dashboard connectivity and health status.
    
    Provides health check information for mobile integration components.
    """
    try:
        redis_client = get_redis()
        
        # Check Redis connectivity
        redis_ping = await redis_client.ping()
        
        # Get active agents count
        active_agents = await get_active_agents_status()
        agent_count = len(active_agents) if active_agents else 0
        
        # Check for recent Claude sessions
        session_keys = await redis_client.keys("claude:session:*")
        active_sessions = len(session_keys)
        
        # Get recent notifications count
        notification_keys = await redis_client.keys("notification:*")
        recent_notifications = len(notification_keys)
        
        status = {
            "success": True,
            "mobile_dashboard_healthy": True,
            "redis_connected": redis_ping,
            "active_agents": agent_count,
            "active_claude_sessions": active_sessions,
            "recent_notifications": recent_notifications,
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "redis_pubsub": redis_ping,
                "agent_coordination": agent_count > 0,
                "notification_system": True,
                "session_management": True
            },
            "endpoints": {
                "session_start": "/api/claude/session-start",
                "session_resume": "/api/claude/session-resume", 
                "mobile_notifications": "/api/mobile/notifications",
                "agent_updates": "/api/mobile/agent-update"
            }
        }
        
        logger.info("📱 Mobile dashboard status check completed", 
                   agents=agent_count,
                   sessions=active_sessions,
                   notifications=recent_notifications)
        
        return status
        
    except Exception as e:
        logger.error("❌ Failed to get mobile dashboard status", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get mobile dashboard status: {str(e)}"
        )


@router.get("/mobile/qr-code")
async def generate_mobile_qr_code(session_id: Optional[str] = None):
    """
    Generate QR code for mobile dashboard access.
    
    Creates QR code for quick mobile device connection to the dashboard.
    """
    try:
        import qrcode
        from io import BytesIO
        import base64
        
        # Construct mobile dashboard URL
        base_url = "http://localhost:8000/dashboard/mobile"
        if session_id:
            dashboard_url = f"{base_url}?session={session_id}"
        else:
            dashboard_url = base_url
        
        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(dashboard_url)
        qr.make(fit=True)
        
        # Create QR code image
        qr_image = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64 for API response
        buffer = BytesIO()
        qr_image.save(buffer, format='PNG')
        qr_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        logger.info("📱 QR code generated for mobile access", 
                   url=dashboard_url,
                   session_id=session_id)
        
        return {
            "success": True,
            "qr_code_base64": f"data:image/png;base64,{qr_base64}",
            "dashboard_url": dashboard_url,
            "session_id": session_id,
            "instructions": "Scan this QR code with your mobile device to access the LeanVibe Agent Hive 2.0 dashboard"
        }
        
    except ImportError:
        logger.warning("QR code generation not available - qrcode library not installed")
        return {
            "success": False,
            "error": "QR code generation not available",
            "dashboard_url": f"http://localhost:8000/dashboard/mobile" + (f"?session={session_id}" if session_id else ""),
            "manual_access": "Navigate to the dashboard URL manually on your mobile device"
        }
    except Exception as e:
        logger.error("❌ Failed to generate QR code", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate QR code: {str(e)}"
        )