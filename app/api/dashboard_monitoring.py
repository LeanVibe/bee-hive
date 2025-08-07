"""
Dashboard Monitoring API for Multi-Agent Coordination System

Provides comprehensive API endpoints for monitoring and controlling the 
multi-agent coordination system with real-time data and recovery controls.

CRITICAL: Addresses 20% coordination success rate with monitoring and control APIs.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
import structlog

from fastapi import APIRouter, HTTPException, Query, Path, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import get_async_session
from ..core.redis import get_redis, get_message_broker
from ..core.enhanced_coordination_bridge import get_enhanced_coordination_metrics
from ..models.agent import Agent, AgentStatus
from ..models.task import Task, TaskStatus, TaskPriority
from ..schemas.agent import AgentResponse
from ..schemas.task import TaskResponse

logger = structlog.get_logger()
router = APIRouter(prefix="/api/dashboard", tags=["dashboard-monitoring"])


@dataclass
class CoordinationMetrics:
    """Real-time coordination success rate and metrics."""
    success_rate: float
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    pending_tasks: int
    average_completion_time: float
    coordination_errors: int
    last_coordination_failure: Optional[str]
    trend_direction: str  # "improving", "declining", "stable"
    

@dataclass  
class AgentHealthData:
    """Comprehensive agent health information."""
    agent_id: str
    name: str
    status: str
    last_heartbeat: Optional[str]
    response_time_ms: float
    task_success_rate: float
    current_task: Optional[str]
    error_count: int
    memory_usage: float
    context_utilization: float
    health_score: float  # 0-100


@dataclass
class TaskDistributionData:
    """Task distribution and queue information."""
    queue_length: int
    tasks_by_status: Dict[str, int]
    tasks_by_priority: Dict[str, int]
    average_wait_time: float
    distribution_efficiency: float
    failed_assignments: int
    reassignments_needed: int


class DashboardWebSocketManager:
    """Manages WebSocket connections for real-time dashboard updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        
    async def connect(self, websocket: WebSocket, connection_id: str, metadata: Dict[str, Any] = None):
        """Connect a new WebSocket client."""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        self.connection_metadata[connection_id] = metadata or {}
        logger.info("Dashboard WebSocket connected", connection_id=connection_id)
        
    def disconnect(self, connection_id: str):
        """Disconnect a WebSocket client."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            del self.connection_metadata[connection_id]
        logger.info("Dashboard WebSocket disconnected", connection_id=connection_id)
        
    async def broadcast(self, message_type: str, data: Dict[str, Any]):
        """Broadcast data to all connected clients."""
        if not self.active_connections:
            return
            
        message = {
            "type": message_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        disconnected = []
        for connection_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.warning("Failed to send WebSocket message", 
                             connection_id=connection_id, error=str(e))
                disconnected.append(connection_id)
        
        # Clean up disconnected clients
        for connection_id in disconnected:
            self.disconnect(connection_id)


# Global WebSocket manager
websocket_manager = DashboardWebSocketManager()


async def get_coordination_metrics(db: AsyncSession) -> CoordinationMetrics:
    """Calculate real-time coordination success metrics."""
    try:
        # Get task statistics for last 24 hours
        since = datetime.utcnow() - timedelta(days=1)
        
        # Total tasks in last 24 hours
        total_tasks_result = await db.execute(
            select(func.count(Task.id)).where(Task.created_at >= since)
        )
        total_tasks = total_tasks_result.scalar() or 0
        
        # Successful tasks
        successful_tasks_result = await db.execute(
            select(func.count(Task.id)).where(
                and_(Task.created_at >= since, Task.status == TaskStatus.COMPLETED)
            )
        )
        successful_tasks = successful_tasks_result.scalar() or 0
        
        # Failed tasks  
        failed_tasks_result = await db.execute(
            select(func.count(Task.id)).where(
                and_(Task.created_at >= since, Task.status == TaskStatus.FAILED)
            )
        )
        failed_tasks = failed_tasks_result.scalar() or 0
        
        # Pending tasks
        pending_tasks_result = await db.execute(
            select(func.count(Task.id)).where(Task.status.in_([TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]))
        )
        pending_tasks = pending_tasks_result.scalar() or 0
        
        # Calculate success rate
        success_rate = (successful_tasks / max(1, total_tasks - pending_tasks)) * 100
        
        # Average completion time for completed tasks  
        completed_tasks = await db.execute(
            select(Task).where(
                and_(
                    Task.status == TaskStatus.COMPLETED,
                    Task.started_at.isnot(None),
                    Task.completed_at.isnot(None),
                    Task.created_at >= since
                )
            )
        )
        
        completion_times = []
        for task in completed_tasks.scalars():
            if task.started_at and task.completed_at:
                duration = (task.completed_at - task.started_at).total_seconds() / 60
                completion_times.append(duration)
        
        average_completion_time = sum(completion_times) / len(completion_times) if completion_times else 0.0
        
        # Determine trend (simplified - compare to previous period)
        prev_since = datetime.utcnow() - timedelta(days=2)
        prev_end = datetime.utcnow() - timedelta(days=1)
        
        prev_total_result = await db.execute(
            select(func.count(Task.id)).where(
                and_(Task.created_at >= prev_since, Task.created_at < prev_end)
            )
        )
        prev_total = prev_total_result.scalar() or 0
        
        prev_successful_result = await db.execute(
            select(func.count(Task.id)).where(
                and_(
                    Task.created_at >= prev_since,
                    Task.created_at < prev_end, 
                    Task.status == TaskStatus.COMPLETED
                )
            )
        )
        prev_successful = prev_successful_result.scalar() or 0
        
        prev_success_rate = (prev_successful / max(1, prev_total)) * 100
        
        if success_rate > prev_success_rate + 5:
            trend_direction = "improving"
        elif success_rate < prev_success_rate - 5:
            trend_direction = "declining"
        else:
            trend_direction = "stable"
            
        # Get last coordination failure
        last_failure_task = await db.execute(
            select(Task).where(Task.status == TaskStatus.FAILED)
            .order_by(Task.updated_at.desc()).limit(1)
        )
        last_failure = last_failure_task.scalar_one_or_none()
        last_coordination_failure = last_failure.error_message if last_failure else None
        
        return CoordinationMetrics(
            success_rate=success_rate,
            total_tasks=total_tasks,
            successful_tasks=successful_tasks,
            failed_tasks=failed_tasks,
            pending_tasks=pending_tasks,
            average_completion_time=average_completion_time,
            coordination_errors=failed_tasks,
            last_coordination_failure=last_coordination_failure,
            trend_direction=trend_direction
        )
        
    except Exception as e:
        logger.error("Failed to calculate coordination metrics", error=str(e))
        return CoordinationMetrics(
            success_rate=0.0,
            total_tasks=0,
            successful_tasks=0,
            failed_tasks=0,
            pending_tasks=0,
            average_completion_time=0.0,
            coordination_errors=0,
            last_coordination_failure=f"Error calculating metrics: {str(e)}",
            trend_direction="unknown"
        )


async def get_agent_health_data(db: AsyncSession) -> List[AgentHealthData]:
    """Get comprehensive health data for all agents."""
    try:
        agents_result = await db.execute(select(Agent))
        agents = agents_result.scalars().all()
        
        health_data = []
        for agent in agents:
            # Calculate response time (mock for now)
            response_time_ms = 150.0 if agent.status == AgentStatus.active else 500.0
            
            # Calculate task success rate
            agent_tasks_result = await db.execute(
                select(func.count(Task.id)).where(Task.assigned_agent_id == agent.id)
            )
            total_agent_tasks = agent_tasks_result.scalar() or 0
            
            successful_agent_tasks_result = await db.execute(
                select(func.count(Task.id)).where(
                    and_(Task.assigned_agent_id == agent.id, Task.status == TaskStatus.COMPLETED)
                )
            )
            successful_agent_tasks = successful_agent_tasks_result.scalar() or 0
            
            task_success_rate = (successful_agent_tasks / max(1, total_agent_tasks)) * 100
            
            # Get current task
            current_task_result = await db.execute(
                select(Task.title).where(
                    and_(
                        Task.assigned_agent_id == agent.id,
                        Task.status == TaskStatus.IN_PROGRESS
                    )
                ).limit(1)
            )
            current_task = current_task_result.scalar_one_or_none()
            
            # Calculate error count (tasks failed)
            error_count_result = await db.execute(
                select(func.count(Task.id)).where(
                    and_(Task.assigned_agent_id == agent.id, Task.status == TaskStatus.FAILED)
                )
            )
            error_count = error_count_result.scalar() or 0
            
            # Calculate health score
            health_score = min(100, (
                (100 if agent.status == AgentStatus.active else 20) * 0.3 +
                task_success_rate * 0.4 +
                (100 if response_time_ms < 200 else max(0, 100 - (response_time_ms - 200) / 10)) * 0.3
            ))
            
            health_data.append(AgentHealthData(
                agent_id=str(agent.id),
                name=agent.name,
                status=agent.status.value,
                last_heartbeat=agent.last_heartbeat.isoformat() if agent.last_heartbeat else None,
                response_time_ms=response_time_ms,
                task_success_rate=task_success_rate,
                current_task=current_task,
                error_count=error_count,
                memory_usage=float(agent.context_window_usage or 0.0),
                context_utilization=float(agent.context_window_usage or 0.0),
                health_score=health_score
            ))
            
        return health_data
        
    except Exception as e:
        logger.error("Failed to get agent health data", error=str(e))
        return []


async def get_task_distribution_data(db: AsyncSession) -> TaskDistributionData:
    """Get task distribution and queue information."""
    try:
        # Queue length (pending + assigned tasks)
        queue_length_result = await db.execute(
            select(func.count(Task.id)).where(
                Task.status.in_([TaskStatus.PENDING, TaskStatus.ASSIGNED])
            )
        )
        queue_length = queue_length_result.scalar() or 0
        
        # Tasks by status
        status_results = await db.execute(
            select(Task.status, func.count(Task.id))
            .group_by(Task.status)
        )
        tasks_by_status = {status.value: count for status, count in status_results.all()}
        
        # Tasks by priority
        priority_results = await db.execute(
            select(Task.priority, func.count(Task.id))
            .group_by(Task.priority)
        )
        tasks_by_priority = {priority.name.lower(): count for priority, count in priority_results.all()}
        
        # Average wait time (time from creation to assignment)
        wait_time_tasks = await db.execute(
            select(Task).where(
                and_(
                    Task.assigned_at.isnot(None),
                    Task.created_at.isnot(None)
                )
            ).limit(100)
        )
        
        wait_times = []
        for task in wait_time_tasks.scalars():
            if task.created_at and task.assigned_at:
                wait_time = (task.assigned_at - task.created_at).total_seconds() / 60
                wait_times.append(wait_time)
        
        average_wait_time = sum(wait_times) / len(wait_times) if wait_times else 0.0
        
        # Distribution efficiency (percentage of tasks assigned vs pending)
        assigned_tasks = tasks_by_status.get("assigned", 0)
        pending_tasks = tasks_by_status.get("pending", 0)
        total_unprocessed = assigned_tasks + pending_tasks
        distribution_efficiency = (assigned_tasks / max(1, total_unprocessed)) * 100
        
        # Failed assignments and reassignments (approximated by retry count)
        failed_assignments_result = await db.execute(
            select(func.count(Task.id)).where(Task.retry_count > 0)
        )
        failed_assignments = failed_assignments_result.scalar() or 0
        
        reassignments_result = await db.execute(
            select(func.count(Task.id)).where(Task.retry_count > 1)
        )
        reassignments_needed = reassignments_result.scalar() or 0
        
        return TaskDistributionData(
            queue_length=queue_length,
            tasks_by_status=tasks_by_status,
            tasks_by_priority=tasks_by_priority,
            average_wait_time=average_wait_time,
            distribution_efficiency=distribution_efficiency,
            failed_assignments=failed_assignments,
            reassignments_needed=reassignments_needed
        )
        
    except Exception as e:
        logger.error("Failed to get task distribution data", error=str(e))
        return TaskDistributionData(
            queue_length=0,
            tasks_by_status={},
            tasks_by_priority={},
            average_wait_time=0.0,
            distribution_efficiency=0.0,
            failed_assignments=0,
            reassignments_needed=0
        )


# ==================== AGENT STATUS & HEALTH APIs ====================

@router.get("/agents/status", response_model=Dict[str, Any])
async def get_agents_status(
    include_inactive: bool = Query(False, description="Include inactive agents"),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get real-time status of all agents with health indicators.
    
    Returns comprehensive agent status including health scores,
    performance metrics, and current task assignments.
    """
    try:
        # Build query based on filters
        query = select(Agent)
        if not include_inactive:
            query = query.where(Agent.status != AgentStatus.inactive)
        
        agents_result = await db.execute(query.order_by(Agent.name))
        agents = agents_result.scalars().all()
        
        # Get health data for all agents
        health_data = await get_agent_health_data(db)
        health_map = {hd.agent_id: hd for hd in health_data}
        
        agents_status = []
        for agent in agents:
            health_info = health_map.get(str(agent.id))
            agent_data = agent.to_dict()
            
            if health_info:
                agent_data.update({
                    "health_score": health_info.health_score,
                    "response_time_ms": health_info.response_time_ms,
                    "task_success_rate": health_info.task_success_rate,
                    "current_task": health_info.current_task,
                    "error_count": health_info.error_count
                })
            
            agents_status.append(agent_data)
        
        return {
            "agents": agents_status,
            "total_agents": len(agents_status),
            "active_agents": len([a for a in agents_status if a["status"] == "active"]),
            "health_summary": {
                "healthy": len([a for a in agents_status if a.get("health_score", 0) > 80]),
                "degraded": len([a for a in agents_status if 50 <= a.get("health_score", 0) <= 80]),
                "unhealthy": len([a for a in agents_status if a.get("health_score", 0) < 50])
            },
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get agents status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to retrieve agent status: {str(e)}")


@router.get("/agents/{agent_id}/metrics", response_model=Dict[str, Any])
async def get_agent_metrics(
    agent_id: str = Path(..., description="Agent ID"),
    time_range_hours: int = Query(24, ge=1, le=168, description="Time range in hours"),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get detailed performance metrics for a specific agent.
    
    Provides comprehensive performance analytics including task success rates,
    response times, and error patterns.
    """
    try:
        # Validate agent exists
        agent_result = await db.execute(
            select(Agent).where(Agent.id == agent_id)
        )
        agent = agent_result.scalar_one_or_none()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Get time range
        since = datetime.utcnow() - timedelta(hours=time_range_hours)
        
        # Get agent tasks in time range
        tasks_result = await db.execute(
            select(Task).where(
                and_(
                    Task.assigned_agent_id == agent_id,
                    Task.created_at >= since
                )
            ).order_by(Task.created_at.desc())
        )
        tasks = tasks_result.scalars().all()
        
        # Calculate metrics
        total_tasks = len(tasks)
        completed_tasks = len([t for t in tasks if t.status == TaskStatus.COMPLETED])
        failed_tasks = len([t for t in tasks if t.status == TaskStatus.FAILED])
        in_progress_tasks = len([t for t in tasks if t.status == TaskStatus.IN_PROGRESS])
        
        success_rate = (completed_tasks / max(1, total_tasks)) * 100
        
        # Calculate average completion time
        completion_times = []
        for task in tasks:
            if task.status == TaskStatus.COMPLETED and task.started_at and task.completed_at:
                duration = (task.completed_at - task.started_at).total_seconds() / 60
                completion_times.append(duration)
        
        avg_completion_time = sum(completion_times) / len(completion_times) if completion_times else 0.0
        
        # Recent task history
        recent_tasks = [
            {
                "id": str(task.id),
                "title": task.title,
                "status": task.status.value,
                "type": task.task_type.value if task.task_type else None,
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "error_message": task.error_message
            }
            for task in tasks[:10]  # Last 10 tasks
        ]
        
        return {
            "agent": agent.to_dict(),
            "time_range_hours": time_range_hours,
            "metrics": {
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "in_progress_tasks": in_progress_tasks,
                "success_rate": success_rate,
                "average_completion_time_minutes": avg_completion_time,
                "error_rate": (failed_tasks / max(1, total_tasks)) * 100
            },
            "recent_tasks": recent_tasks,
            "performance_trends": {
                "hourly_task_completion": [],  # Could be populated with hourly breakdown
                "error_patterns": [],  # Could analyze error message patterns
                "capacity_utilization": float(agent.context_window_usage or 0.0)
            },
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get agent metrics", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to retrieve agent metrics: {str(e)}")


@router.post("/agents/{agent_id}/restart", response_model=Dict[str, Any])
async def restart_agent(
    agent_id: str = Path(..., description="Agent ID"),
    force: bool = Query(False, description="Force restart even if agent has active tasks"),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Restart a specific agent with optional force parameter.
    
    Safely restarts an agent, optionally reassigning active tasks to other agents.
    """
    try:
        # Validate agent exists  
        agent_result = await db.execute(
            select(Agent).where(Agent.id == agent_id)
        )
        agent = agent_result.scalar_one_or_none()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Check for active tasks
        active_tasks_result = await db.execute(
            select(func.count(Task.id)).where(
                and_(
                    Task.assigned_agent_id == agent_id,
                    Task.status == TaskStatus.IN_PROGRESS
                )
            )
        )
        active_tasks_count = active_tasks_result.scalar() or 0
        
        if active_tasks_count > 0 and not force:
            return {
                "error": "Agent has active tasks. Use force=true to restart anyway.",
                "active_tasks": active_tasks_count,
                "agent_id": agent_id
            }
        
        # Update agent status to trigger restart
        agent.status = AgentStatus.maintenance
        agent.updated_at = datetime.utcnow()
        await db.commit()
        
        # If force restart, reassign active tasks
        if force and active_tasks_count > 0:
            active_tasks_result = await db.execute(
                select(Task).where(
                    and_(
                        Task.assigned_agent_id == agent_id,
                        Task.status == TaskStatus.IN_PROGRESS
                    )
                )
            )
            active_tasks = active_tasks_result.scalars().all()
            
            for task in active_tasks:
                task.status = TaskStatus.PENDING
                task.assigned_agent_id = None
                task.assigned_at = None
                task.error_message = f"Reassigned due to agent {agent.name} restart"
            
            await db.commit()
        
        # Send restart command via Redis
        try:
            message_broker = get_message_broker()
            await message_broker.send_message(
                from_agent="orchestrator",
                to_agent=agent_id,
                message_type="restart_command",
                payload={
                    "force": force,
                    "timestamp": datetime.utcnow().isoformat(),
                    "reason": "Dashboard restart request"
                }
            )
        except Exception as redis_error:
            logger.warning("Failed to send restart command via Redis", error=str(redis_error))
        
        # Broadcast restart event to dashboard clients
        await websocket_manager.broadcast("agent_restart", {
            "agent_id": agent_id,
            "agent_name": agent.name,
            "force": force,
            "active_tasks_reassigned": active_tasks_count if force else 0
        })
        
        return {
            "success": True,
            "agent_id": agent_id,
            "agent_name": agent.name,
            "status": "restart_initiated",
            "active_tasks_reassigned": active_tasks_count if force else 0,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to restart agent", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to restart agent: {str(e)}")


@router.get("/agents/heartbeat", response_model=Dict[str, Any])
async def get_agents_heartbeat(
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get last heartbeat timestamps for all agents.
    
    Provides quick health check information for monitoring systems.
    """
    try:
        agents_result = await db.execute(
            select(Agent.id, Agent.name, Agent.last_heartbeat, Agent.status)
            .order_by(Agent.name)
        )
        agents = agents_result.all()
        
        current_time = datetime.utcnow()
        heartbeat_data = []
        
        for agent_id, name, last_heartbeat, status in agents:
            if last_heartbeat:
                time_since_heartbeat = (current_time - last_heartbeat).total_seconds()
                is_stale = time_since_heartbeat > 300  # 5 minutes
            else:
                time_since_heartbeat = None
                is_stale = True
            
            heartbeat_data.append({
                "agent_id": str(agent_id),
                "name": name,
                "status": status.value,
                "last_heartbeat": last_heartbeat.isoformat() if last_heartbeat else None,
                "seconds_since_heartbeat": time_since_heartbeat,
                "is_stale": is_stale,
                "health_status": "healthy" if not is_stale and status == AgentStatus.active else "degraded"
            })
        
        # Summary statistics
        total_agents = len(heartbeat_data)
        healthy_agents = len([a for a in heartbeat_data if a["health_status"] == "healthy"])
        stale_heartbeats = len([a for a in heartbeat_data if a["is_stale"]])
        
        return {
            "agents": heartbeat_data,
            "summary": {
                "total_agents": total_agents,
                "healthy_agents": healthy_agents,
                "stale_heartbeats": stale_heartbeats,
                "overall_health": "healthy" if stale_heartbeats == 0 else "degraded"
            },
            "last_updated": current_time.isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get agent heartbeats", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to retrieve agent heartbeats: {str(e)}")


# ==================== COORDINATION MONITORING APIs ====================

@router.get("/coordination/success-rate", response_model=Dict[str, Any])
async def get_coordination_success_rate(
    time_range_hours: int = Query(24, ge=1, le=168, description="Time range in hours"),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get live coordination success rate with trends and detailed breakdown.
    
    Critical endpoint for monitoring the 20% success rate issue.
    """
    try:
        metrics = await get_coordination_metrics(db)
        
        # Get hourly breakdown for trend analysis
        hours_ago = time_range_hours
        hourly_data = []
        
        for hour in range(hours_ago):
            hour_start = datetime.utcnow() - timedelta(hours=hour+1)
            hour_end = datetime.utcnow() - timedelta(hours=hour)
            
            hour_tasks_result = await db.execute(
                select(func.count(Task.id)).where(
                    and_(Task.created_at >= hour_start, Task.created_at < hour_end)
                )
            )
            hour_total = hour_tasks_result.scalar() or 0
            
            hour_success_result = await db.execute(
                select(func.count(Task.id)).where(
                    and_(
                        Task.created_at >= hour_start,
                        Task.created_at < hour_end,
                        Task.status == TaskStatus.COMPLETED
                    )
                )
            )
            hour_success = hour_success_result.scalar() or 0
            
            hourly_data.append({
                "hour": hour_start.strftime("%H:%M"),
                "total_tasks": hour_total,
                "successful_tasks": hour_success,
                "success_rate": (hour_success / max(1, hour_total)) * 100
            })
        
        # Identify failure patterns
        failure_patterns = {}
        failed_tasks_result = await db.execute(
            select(Task.error_message).where(
                and_(
                    Task.status == TaskStatus.FAILED,
                    Task.error_message.isnot(None),
                    Task.created_at >= datetime.utcnow() - timedelta(hours=time_range_hours)
                )
            )
        )
        
        for error_message in failed_tasks_result.scalars():
            if error_message:
                # Simplified pattern extraction
                pattern = error_message.split(':')[0] if ':' in error_message else error_message[:50]
                failure_patterns[pattern] = failure_patterns.get(pattern, 0) + 1
        
        return {
            "current_metrics": asdict(metrics),
            "time_range_hours": time_range_hours,
            "hourly_breakdown": hourly_data,
            "failure_patterns": failure_patterns,
            "critical_alerts": [
                {
                    "level": "critical" if metrics.success_rate < 50 else "warning" if metrics.success_rate < 80 else "info",
                    "message": f"Coordination success rate is {metrics.success_rate:.1f}%",
                    "threshold": "Expected >90%",
                    "impact": "Core autonomous development functionality affected"
                }
            ],
            "recommendations": [
                "Check Redis connectivity and message serialization" if metrics.success_rate < 30 else None,
                "Review agent assignment logic" if metrics.failed_tasks > metrics.successful_tasks else None,
                "Investigate task timeout configurations" if metrics.average_completion_time > 30 else None
            ],
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get coordination success rate", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to retrieve coordination success rate: {str(e)}")


@router.get("/coordination/failures", response_model=Dict[str, Any])
async def get_coordination_failures(
    limit: int = Query(50, ge=1, le=200, description="Number of recent failures to return"),
    include_resolved: bool = Query(False, description="Include resolved failures"),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get detailed failure analysis with breakdown by type and severity.
    
    Provides actionable insights for resolving coordination issues.
    """
    try:
        # Get recent failed tasks
        query = select(Task).where(Task.status == TaskStatus.FAILED)
        
        if not include_resolved:
            # Only show unresolved failures (could add resolution tracking later)
            query = query.where(Task.retry_count < Task.max_retries)
        
        failures_result = await db.execute(
            query.order_by(Task.updated_at.desc()).limit(limit)
        )
        failures = failures_result.scalars().all()
        
        # Analyze failure patterns
        failure_analysis = {
            "by_type": {},
            "by_agent": {},
            "by_error_pattern": {},
            "by_hour": {}
        }
        
        detailed_failures = []
        current_time = datetime.utcnow()
        
        for task in failures:
            # Categorize by type
            task_type = task.task_type.value if task.task_type else "unknown"
            failure_analysis["by_type"][task_type] = failure_analysis["by_type"].get(task_type, 0) + 1
            
            # Categorize by agent
            agent_id = str(task.assigned_agent_id) if task.assigned_agent_id else "unassigned"
            failure_analysis["by_agent"][agent_id] = failure_analysis["by_agent"].get(agent_id, 0) + 1
            
            # Categorize by error pattern
            if task.error_message:
                pattern = task.error_message.split(':')[0] if ':' in task.error_message else "generic_error"
                failure_analysis["by_error_pattern"][pattern] = failure_analysis["by_error_pattern"].get(pattern, 0) + 1
            
            # Categorize by hour
            if task.updated_at:
                hour = task.updated_at.strftime("%H:00")
                failure_analysis["by_hour"][hour] = failure_analysis["by_hour"].get(hour, 0) + 1
            
            # Add to detailed failures
            time_since_failure = None
            if task.updated_at:
                time_since_failure = (current_time - task.updated_at).total_seconds() / 60
            
            detailed_failures.append({
                "task_id": str(task.id),
                "title": task.title,
                "type": task_type,
                "assigned_agent_id": str(task.assigned_agent_id) if task.assigned_agent_id else None,
                "error_message": task.error_message,
                "retry_count": task.retry_count,
                "max_retries": task.max_retries,
                "can_retry": task.retry_count < task.max_retries,
                "priority": task.priority.name.lower(),
                "failed_at": task.updated_at.isoformat() if task.updated_at else None,
                "minutes_since_failure": time_since_failure,
                "recovery_actions": [
                    "retry_task",
                    "reassign_agent", 
                    "escalate_priority"
                ]
            })
        
        # Identify top failure causes
        top_error_patterns = sorted(
            failure_analysis["by_error_pattern"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "total_failures": len(failures),
            "analysis": failure_analysis,
            "top_error_patterns": [{"pattern": p, "count": c} for p, c in top_error_patterns],
            "detailed_failures": detailed_failures,
            "recovery_recommendations": [
                {
                    "action": "Restart affected agents",
                    "applicable_to": len(set(f["assigned_agent_id"] for f in detailed_failures if f["assigned_agent_id"])),
                    "urgency": "high" if len(detailed_failures) > 10 else "medium"
                },
                {
                    "action": "Increase task timeout values",
                    "applicable_to": len([f for f in detailed_failures if "timeout" in (f["error_message"] or "").lower()]),
                    "urgency": "medium"
                },
                {
                    "action": "Review Redis connection stability",
                    "applicable_to": len([f for f in detailed_failures if "redis" in (f["error_message"] or "").lower()]),
                    "urgency": "critical" if any("redis" in (f["error_message"] or "").lower() for f in detailed_failures) else "low"
                }
            ],
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get coordination failures", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to retrieve coordination failures: {str(e)}")


@router.post("/coordination/reset", response_model=Dict[str, Any])
async def reset_coordination_system(
    reset_type: str = Query("soft", regex="^(soft|hard|full)$", description="Type of reset to perform"),
    confirm: bool = Query(False, description="Confirmation required for reset"),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Emergency coordination system reset with different severity levels.
    
    Provides recovery mechanisms for when coordination system is failing.
    """
    if not confirm:
        return {
            "error": "Reset confirmation required",
            "message": "Set confirm=true to proceed with coordination reset",
            "reset_type": reset_type
        }
    
    try:
        reset_actions = []
        
        if reset_type in ["soft", "hard", "full"]:
            # Reset stuck tasks to pending
            stuck_tasks_result = await db.execute(
                select(Task).where(
                    and_(
                        Task.status == TaskStatus.IN_PROGRESS,
                        Task.started_at < datetime.utcnow() - timedelta(hours=2)
                    )
                )
            )
            stuck_tasks = stuck_tasks_result.scalars().all()
            
            for task in stuck_tasks:
                task.status = TaskStatus.PENDING
                task.assigned_agent_id = None
                task.error_message = f"Reset by coordination system reset ({reset_type})"
                
            reset_actions.append(f"Reset {len(stuck_tasks)} stuck tasks to pending")
            await db.commit()
        
        if reset_type in ["hard", "full"]:
            # Reset all agents to maintenance then active
            agents_result = await db.execute(select(Agent))
            agents = agents_result.scalars().all()
            
            for agent in agents:
                if agent.status not in [AgentStatus.inactive]:
                    agent.status = AgentStatus.maintenance
                    agent.last_heartbeat = None
                    
            await db.commit()
            reset_actions.append(f"Reset {len(agents)} agents to maintenance mode")
            
            # Send reset commands via Redis
            try:
                message_broker = get_message_broker()
                await message_broker.broadcast_message(
                    from_agent="orchestrator",
                    message_type="coordination_reset",
                    payload={
                        "reset_type": reset_type,
                        "timestamp": datetime.utcnow().isoformat(),
                        "reason": "Dashboard emergency reset"
                    }
                )
                reset_actions.append("Broadcast reset commands via Redis")
            except Exception as redis_error:
                logger.warning("Failed to broadcast reset commands", error=str(redis_error))
                reset_actions.append(f"Redis broadcast failed: {str(redis_error)}")
        
        if reset_type == "full":
            # Clear Redis streams (this would need Redis access)
            try:
                redis_client = get_redis()
                # Delete agent message streams
                keys = await redis_client.keys("agent_messages:*")
                if keys:
                    await redis_client.delete(*keys)
                    reset_actions.append(f"Cleared {len(keys)} Redis message streams")
                
                # Clear workflow coordination data
                workflow_keys = await redis_client.keys("workflow_coordination:*")
                if workflow_keys:
                    await redis_client.delete(*workflow_keys)
                    reset_actions.append(f"Cleared {len(workflow_keys)} workflow coordination records")
                    
            except Exception as redis_error:
                logger.warning("Failed to clear Redis data", error=str(redis_error))
                reset_actions.append(f"Redis cleanup failed: {str(redis_error)}")
        
        # Broadcast reset event to dashboard clients
        await websocket_manager.broadcast("coordination_reset", {
            "reset_type": reset_type,
            "actions": reset_actions,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "success": True,
            "reset_type": reset_type,
            "actions_performed": reset_actions,
            "timestamp": datetime.utcnow().isoformat(),
            "next_steps": [
                "Monitor agent health endpoints for recovery",
                "Check coordination success rate in 5 minutes",
                "Review system logs for any persistent issues"
            ]
        }
        
    except Exception as e:
        logger.error("Failed to reset coordination system", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to reset coordination system: {str(e)}")


@router.get("/coordination/diagnostics", response_model=Dict[str, Any])
async def get_coordination_diagnostics(
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get comprehensive diagnostic data for coordination system health.
    
    Provides deep insights into system state for troubleshooting.
    """
    try:
        diagnostics = {}
        
        # Database diagnostics
        db_checks = {}
        
        # Check agent table health
        agents_count_result = await db.execute(select(func.count(Agent.id)))
        db_checks["total_agents"] = agents_count_result.scalar()
        
        active_agents_result = await db.execute(
            select(func.count(Agent.id)).where(Agent.status == AgentStatus.active)
        )
        db_checks["active_agents"] = active_agents_result.scalar()
        
        # Check task table health
        tasks_count_result = await db.execute(select(func.count(Task.id)))
        db_checks["total_tasks"] = tasks_count_result.scalar()
        
        # Check for orphaned tasks (assigned to non-existent agents)
        orphaned_tasks_result = await db.execute(
            select(func.count(Task.id)).where(
                and_(
                    Task.assigned_agent_id.isnot(None),
                    ~Task.assigned_agent_id.in_(select(Agent.id))
                )
            )
        )
        db_checks["orphaned_tasks"] = orphaned_tasks_result.scalar()
        
        diagnostics["database"] = db_checks
        
        # Redis diagnostics
        redis_checks = {}
        try:
            redis_client = get_redis()
            
            # Check Redis connectivity
            ping_result = await redis_client.ping()
            redis_checks["connectivity"] = "healthy" if ping_result else "failed"
            
            # Check message streams
            agent_streams = await redis_client.keys("agent_messages:*")
            redis_checks["active_streams"] = len(agent_streams)
            
            # Check workflow coordination data
            workflow_keys = await redis_client.keys("workflow_coordination:*")
            redis_checks["active_workflows"] = len(workflow_keys)
            
            # Check for stale data
            info = await redis_client.info("memory")
            redis_checks["memory_usage"] = info.get("used_memory_human", "unknown")
            
        except Exception as redis_error:
            redis_checks["error"] = str(redis_error)
            redis_checks["connectivity"] = "failed"
        
        diagnostics["redis"] = redis_checks
        
        # Coordination system diagnostics
        coord_checks = {}
        
        # Check task assignment efficiency
        unassigned_tasks_result = await db.execute(
            select(func.count(Task.id)).where(
                and_(
                    Task.status.in_([TaskStatus.PENDING, TaskStatus.ASSIGNED]),
                    Task.created_at < datetime.utcnow() - timedelta(minutes=10)
                )
            )
        )
        coord_checks["stale_unassigned_tasks"] = unassigned_tasks_result.scalar()
        
        # Check for long-running tasks
        long_running_result = await db.execute(
            select(func.count(Task.id)).where(
                and_(
                    Task.status == TaskStatus.IN_PROGRESS,
                    Task.started_at < datetime.utcnow() - timedelta(hours=1)
                )
            )
        )
        coord_checks["long_running_tasks"] = long_running_result.scalar()
        
        # Check agent heartbeat freshness
        stale_heartbeat_result = await db.execute(
            select(func.count(Agent.id)).where(
                and_(
                    Agent.status == AgentStatus.active,
                    or_(
                        Agent.last_heartbeat.is_(None),
                        Agent.last_heartbeat < datetime.utcnow() - timedelta(minutes=5)
                    )
                )
            )
        )
        coord_checks["agents_with_stale_heartbeats"] = stale_heartbeat_result.scalar()
        
        diagnostics["coordination"] = coord_checks
        
        # System health score calculation
        health_score = 100
        
        # Deduct points for issues
        if coord_checks["stale_unassigned_tasks"] > 0:
            health_score -= min(20, coord_checks["stale_unassigned_tasks"] * 2)
            
        if coord_checks["long_running_tasks"] > 0:
            health_score -= min(15, coord_checks["long_running_tasks"] * 3)
            
        if coord_checks["agents_with_stale_heartbeats"] > 0:
            health_score -= min(25, coord_checks["agents_with_stale_heartbeats"] * 5)
            
        if db_checks["orphaned_tasks"] > 0:
            health_score -= min(30, db_checks["orphaned_tasks"] * 5)
            
        if redis_checks.get("connectivity") != "healthy":
            health_score -= 40
        
        diagnostics["overall_health"] = {
            "score": max(0, health_score),
            "status": "healthy" if health_score >= 90 else "degraded" if health_score >= 70 else "critical",
            "issues": []
        }
        
        # Add issue descriptions
        if coord_checks["stale_unassigned_tasks"] > 0:
            diagnostics["overall_health"]["issues"].append(
                f"{coord_checks['stale_unassigned_tasks']} tasks pending assignment for >10 minutes"
            )
            
        if coord_checks["long_running_tasks"] > 0:
            diagnostics["overall_health"]["issues"].append(
                f"{coord_checks['long_running_tasks']} tasks running for >1 hour"
            )
            
        if coord_checks["agents_with_stale_heartbeats"] > 0:
            diagnostics["overall_health"]["issues"].append(
                f"{coord_checks['agents_with_stale_heartbeats']} agents with stale heartbeats"
            )
        
        return {
            "diagnostics": diagnostics,
            "timestamp": datetime.utcnow().isoformat(),
            "recommendations": [
                "Restart agents with stale heartbeats" if coord_checks["agents_with_stale_heartbeats"] > 0 else None,
                "Investigate long-running tasks" if coord_checks["long_running_tasks"] > 0 else None,
                "Check Redis connectivity" if redis_checks.get("connectivity") != "healthy" else None,
                "Clean up orphaned tasks" if db_checks["orphaned_tasks"] > 0 else None
            ]
        }
        
    except Exception as e:
        logger.error("Failed to get coordination diagnostics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to retrieve coordination diagnostics: {str(e)}")


# ==================== ENHANCED COORDINATION INTEGRATION ====================

@router.get("/coordination/enhanced-metrics", response_model=Dict[str, Any])
async def get_enhanced_coordination_metrics_endpoint():
    """
    Get sophisticated multi-agent coordination metrics from enhanced coordination system.
    
    This endpoint bridges the advanced coordination system with dashboard monitoring,
    showing real autonomous development progress instead of basic agent status.
    
    Returns:
    - Coordination success rates from sophisticated multi-agent collaboration
    - Business value metrics (time saved, productivity gains, ROI)
    - Agent specialization and collaboration patterns  
    - Decision points requiring human developer input
    - Real autonomous development status and progress
    """
    try:
        enhanced_metrics = await get_enhanced_coordination_metrics()
        
        return {
            "status": "success",
            "message": "Enhanced coordination metrics retrieved",
            "data": enhanced_metrics,
            "timestamp": datetime.utcnow().isoformat(),
            "integration_status": "connected",
            "autonomous_development_active": enhanced_metrics.get("autonomous_development_active", False)
        }
        
    except Exception as e:
        logger.error("❌ Failed to get enhanced coordination metrics", error=str(e))
        return {
            "status": "error", 
            "message": "Enhanced coordination system not available",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
            "integration_status": "disconnected",
            "autonomous_development_active": False
        }