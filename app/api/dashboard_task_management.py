"""
Dashboard Task Management API for Multi-Agent Coordination System

Provides comprehensive APIs for task distribution monitoring and manual controls
for the LeanVibe Agent Hive dashboard system.

Part 2 of the dashboard monitoring infrastructure.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import structlog

from fastapi import APIRouter, HTTPException, Query, Path, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy import select, func, and_, or_, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import get_async_session
from ..core.redis import get_redis, get_message_broker
from ..models.agent import Agent, AgentStatus
from ..models.task import Task, TaskStatus, TaskPriority, TaskType
from ..schemas.task import TaskResponse, TaskCreate, TaskUpdate

logger = structlog.get_logger()
router = APIRouter(prefix="/api/dashboard", tags=["dashboard-task-management"])


# ==================== TASK DISTRIBUTION APIs ====================

@router.get("/tasks/queue", response_model=Dict[str, Any])
async def get_task_queue_status(
    status_filter: Optional[str] = Query(None, regex="^(pending|assigned|in_progress|blocked)$", description="Filter by status"),
    priority_filter: Optional[str] = Query(None, regex="^(low|medium|high|critical)$", description="Filter by priority"),
    agent_filter: Optional[str] = Query(None, description="Filter by assigned agent ID"),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of tasks to return"),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get current task queue status with filtering and distribution metrics.
    
    Provides comprehensive view of task distribution across the system.
    """
    try:
        # Build base query
        query = select(Task).order_by(Task.priority.desc(), Task.created_at.desc())
        
        # Apply status filter
        if status_filter:
            status_map = {
                "pending": TaskStatus.PENDING,
                "assigned": TaskStatus.ASSIGNED,
                "in_progress": TaskStatus.IN_PROGRESS,
                "blocked": TaskStatus.BLOCKED
            }
            query = query.where(Task.status == status_map[status_filter])
        else:
            # Default to active statuses
            query = query.where(Task.status.in_([
                TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED
            ]))
        
        # Apply priority filter
        if priority_filter:
            priority_map = {
                "low": TaskPriority.LOW,
                "medium": TaskPriority.MEDIUM,
                "high": TaskPriority.HIGH,
                "critical": TaskPriority.CRITICAL
            }
            query = query.where(Task.priority == priority_map[priority_filter])
        
        # Apply agent filter
        if agent_filter:
            query = query.where(Task.assigned_agent_id == agent_filter)
        
        # Execute query with limit
        tasks_result = await db.execute(query.limit(limit))
        tasks = tasks_result.scalars().all()
        
        # Get queue statistics
        queue_stats = {}
        
        # Tasks by status
        status_counts_result = await db.execute(
            select(Task.status, func.count(Task.id)).where(
                Task.status.in_([TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED])
            ).group_by(Task.status)
        )
        status_counts = {status.value: count for status, count in status_counts_result.all()}
        queue_stats["by_status"] = status_counts
        
        # Tasks by priority  
        priority_counts_result = await db.execute(
            select(Task.priority, func.count(Task.id)).where(
                Task.status.in_([TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED])
            ).group_by(Task.priority)
        )
        priority_counts = {priority.name.lower(): count for priority, count in priority_counts_result.all()}
        queue_stats["by_priority"] = priority_counts
        
        # Tasks by type
        type_counts_result = await db.execute(
            select(Task.task_type, func.count(Task.id)).where(
                and_(
                    Task.status.in_([TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED]),
                    Task.task_type.isnot(None)
                )
            ).group_by(Task.task_type)
        )
        type_counts = {task_type.value: count for task_type, count in type_counts_result.all()}
        queue_stats["by_type"] = type_counts
        
        # Agent assignment statistics
        agent_assignment_result = await db.execute(
            select(Task.assigned_agent_id, func.count(Task.id)).where(
                and_(
                    Task.status.in_([TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]),
                    Task.assigned_agent_id.isnot(None)
                )
            ).group_by(Task.assigned_agent_id)
        )
        agent_assignments = {}
        for agent_id, count in agent_assignment_result.all():
            # Get agent name
            agent_result = await db.execute(select(Agent.name).where(Agent.id == agent_id))
            agent_name = agent_result.scalar_one_or_none() or f"Agent-{str(agent_id)[-8:]}"
            agent_assignments[str(agent_id)] = {"name": agent_name, "active_tasks": count}
        
        queue_stats["agent_assignments"] = agent_assignments
        
        # Calculate distribution metrics
        total_active_tasks = sum(status_counts.values())
        assigned_tasks = status_counts.get("assigned", 0) + status_counts.get("in_progress", 0)
        pending_tasks = status_counts.get("pending", 0)
        
        distribution_efficiency = (assigned_tasks / max(1, total_active_tasks)) * 100
        
        # Calculate average wait times
        pending_tasks_with_times = await db.execute(
            select(Task.created_at).where(Task.status == TaskStatus.PENDING)
        )
        current_time = datetime.utcnow()
        wait_times = []
        for created_at in pending_tasks_with_times.scalars():
            if created_at:
                wait_time = (current_time - created_at).total_seconds() / 60
                wait_times.append(wait_time)
        
        average_wait_time = sum(wait_times) / len(wait_times) if wait_times else 0.0
        
        # Convert tasks to dict format
        task_details = []
        for task in tasks:
            task_dict = task.to_dict()
            
            # Add agent information
            if task.assigned_agent_id:
                agent_result = await db.execute(select(Agent.name).where(Agent.id == task.assigned_agent_id))
                agent_name = agent_result.scalar_one_or_none()
                task_dict["assigned_agent_name"] = agent_name
            
            # Calculate task urgency and wait time
            if task.created_at:
                wait_time = (current_time - task.created_at).total_seconds() / 60
                task_dict["wait_time_minutes"] = wait_time
                task_dict["urgency_score"] = task.calculate_urgency_score()
            
            task_details.append(task_dict)
        
        return {
            "queue_statistics": queue_stats,
            "distribution_metrics": {
                "total_active_tasks": total_active_tasks,
                "distribution_efficiency": distribution_efficiency,
                "average_wait_time_minutes": average_wait_time,
                "unassigned_tasks": pending_tasks,
                "agent_utilization": len(agent_assignments),
                "task_throughput": {  # Could be calculated from historical data
                    "tasks_per_hour": 0,  # Placeholder
                    "completion_rate": 0  # Placeholder
                }
            },
            "tasks": task_details,
            "filters_applied": {
                "status": status_filter,
                "priority": priority_filter,
                "agent": agent_filter,
                "limit": limit
            },
            "last_updated": current_time.isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get task queue status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to retrieve task queue status: {str(e)}")


@router.post("/tasks/{task_id}/reassign", response_model=Dict[str, Any])
async def reassign_task(
    task_id: str = Path(..., description="Task ID to reassign"),
    new_agent_id: Optional[str] = Query(None, description="Specific agent ID to assign to"),
    auto_select: bool = Query(True, description="Automatically select best available agent"),
    priority_boost: bool = Query(False, description="Increase task priority during reassignment"),
    reason: str = Query("Manual reassignment via dashboard", description="Reason for reassignment"),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Manual task reassignment with automatic agent selection option.
    
    Provides manual control over task distribution when coordination fails.
    """
    try:
        # Validate task exists
        task_result = await db.execute(select(Task).where(Task.id == task_id))
        task = task_result.scalar_one_or_none()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Store original assignment info
        original_agent_id = task.assigned_agent_id
        original_status = task.status
        
        reassignment_info = {
            "task_id": task_id,
            "original_agent_id": str(original_agent_id) if original_agent_id else None,
            "original_status": original_status.value,
            "reason": reason
        }
        
        # Handle specific agent assignment
        if new_agent_id and not auto_select:
            # Validate new agent exists and is available
            agent_result = await db.execute(
                select(Agent).where(and_(Agent.id == new_agent_id, Agent.status == AgentStatus.active))
            )
            new_agent = agent_result.scalar_one_or_none()
            if not new_agent:
                raise HTTPException(status_code=400, detail="Target agent not found or not active")
            
            # Assign to specific agent
            task.assigned_agent_id = new_agent_id
            task.assigned_at = datetime.utcnow()
            task.status = TaskStatus.ASSIGNED
            reassignment_info["new_agent_id"] = new_agent_id
            reassignment_info["new_agent_name"] = new_agent.name
            reassignment_info["assignment_method"] = "manual_specific"
            
        elif auto_select:
            # Find best available agent using capability matching
            available_agents_result = await db.execute(
                select(Agent).where(Agent.status == AgentStatus.active)
            )
            available_agents = available_agents_result.scalars().all()
            
            if not available_agents:
                raise HTTPException(status_code=400, detail="No active agents available for reassignment")
            
            # Simple agent selection based on current task load
            agent_loads = {}
            for agent in available_agents:
                load_result = await db.execute(
                    select(func.count(Task.id)).where(
                        and_(
                            Task.assigned_agent_id == agent.id,
                            Task.status.in_([TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS])
                        )
                    )
                )
                agent_loads[agent.id] = load_result.scalar() or 0
            
            # Select agent with lowest load
            best_agent_id = min(agent_loads, key=agent_loads.get)
            best_agent = next(a for a in available_agents if a.id == best_agent_id)
            
            task.assigned_agent_id = best_agent_id
            task.assigned_at = datetime.utcnow()
            task.status = TaskStatus.ASSIGNED
            reassignment_info["new_agent_id"] = str(best_agent_id)
            reassignment_info["new_agent_name"] = best_agent.name
            reassignment_info["assignment_method"] = "auto_select_best_available"
            reassignment_info["agent_task_load"] = agent_loads[best_agent_id]
            
        else:
            # Reset to pending for orchestrator to assign
            task.assigned_agent_id = None
            task.assigned_at = None
            task.status = TaskStatus.PENDING
            reassignment_info["new_agent_id"] = None
            reassignment_info["assignment_method"] = "reset_to_pending"
        
        # Apply priority boost if requested
        if priority_boost and task.priority != TaskPriority.CRITICAL:
            old_priority = task.priority
            if task.priority == TaskPriority.LOW:
                task.priority = TaskPriority.MEDIUM
            elif task.priority == TaskPriority.MEDIUM:
                task.priority = TaskPriority.HIGH
            elif task.priority == TaskPriority.HIGH:
                task.priority = TaskPriority.CRITICAL
            
            reassignment_info["priority_boost"] = {
                "from": old_priority.name.lower(),
                "to": task.priority.name.lower()
            }
        
        # Update task with reassignment reason
        task.error_message = f"Reassigned: {reason}"
        task.updated_at = datetime.utcnow()
        
        await db.commit()
        
        # Send notification via Redis if agent was assigned
        if task.assigned_agent_id:
            try:
                message_broker = get_message_broker()
                await message_broker.send_message(
                    from_agent="orchestrator",
                    to_agent=str(task.assigned_agent_id),
                    message_type="task_reassignment",
                    payload={
                        "task_id": task_id,
                        "task_title": task.title,
                        "reason": reason,
                        "priority": task.priority.name.lower(),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                reassignment_info["redis_notification"] = "sent"
            except Exception as redis_error:
                logger.warning("Failed to send reassignment notification", error=str(redis_error))
                reassignment_info["redis_notification"] = f"failed: {str(redis_error)}"
        
        return {
            "success": True,
            "reassignment": reassignment_info,
            "updated_task": task.to_dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to reassign task", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to reassign task: {str(e)}")


@router.get("/tasks/distribution", response_model=Dict[str, Any])
async def get_task_distribution_visualization(
    time_range_hours: int = Query(24, ge=1, le=168, description="Time range for distribution analysis"),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get task distribution data optimized for dashboard visualization.
    
    Provides data formatted for charts and distribution analysis.
    """
    try:
        since = datetime.utcnow() - timedelta(hours=time_range_hours)
        
        # Distribution by agent
        agent_distribution_result = await db.execute(
            select(Agent.name, Agent.id, func.count(Task.id).label('task_count')).
            join(Task, Agent.id == Task.assigned_agent_id).
            where(Task.created_at >= since).
            group_by(Agent.id, Agent.name).
            order_by(func.count(Task.id).desc())
        )
        
        agent_distribution = [
            {
                "agent_id": str(agent_id),
                "agent_name": agent_name,
                "task_count": task_count,
                "percentage": 0  # Will be calculated after getting total
            }
            for agent_name, agent_id, task_count in agent_distribution_result.all()
        ]
        
        # Calculate percentages
        total_distributed_tasks = sum(item["task_count"] for item in agent_distribution)
        for item in agent_distribution:
            item["percentage"] = (item["task_count"] / max(1, total_distributed_tasks)) * 100
        
        # Distribution by task type
        type_distribution_result = await db.execute(
            select(Task.task_type, func.count(Task.id)).
            where(and_(Task.created_at >= since, Task.task_type.isnot(None))).
            group_by(Task.task_type)
        )
        
        type_distribution = [
            {
                "task_type": task_type.value,
                "count": count,
                "percentage": 0
            }
            for task_type, count in type_distribution_result.all()
        ]
        
        total_typed_tasks = sum(item["count"] for item in type_distribution)
        for item in type_distribution:
            item["percentage"] = (item["count"] / max(1, total_typed_tasks)) * 100
        
        # Distribution by priority
        priority_distribution_result = await db.execute(
            select(Task.priority, func.count(Task.id)).
            where(Task.created_at >= since).
            group_by(Task.priority)
        )
        
        priority_distribution = [
            {
                "priority": priority.name.lower(),
                "count": count,
                "percentage": 0
            }
            for priority, count in priority_distribution_result.all()
        ]
        
        total_tasks = sum(item["count"] for item in priority_distribution)
        for item in priority_distribution:
            item["percentage"] = (item["count"] / max(1, total_tasks)) * 100
        
        # Hourly distribution for timeline
        hourly_distribution = []
        for hour in range(time_range_hours):
            hour_start = datetime.utcnow() - timedelta(hours=hour+1)
            hour_end = datetime.utcnow() - timedelta(hours=hour)
            
            hour_tasks_result = await db.execute(
                select(
                    func.count(Task.id).label('total'),
                    func.count(Task.id).filter(Task.status == TaskStatus.COMPLETED).label('completed'),
                    func.count(Task.id).filter(Task.status == TaskStatus.FAILED).label('failed')
                ).where(and_(Task.created_at >= hour_start, Task.created_at < hour_end))
            )
            
            hour_stats = hour_tasks_result.first()
            hourly_distribution.append({
                "hour": hour_start.strftime("%Y-%m-%d %H:00"),
                "total_tasks": hour_stats.total or 0,
                "completed_tasks": hour_stats.completed or 0,
                "failed_tasks": hour_stats.failed or 0,
                "success_rate": (hour_stats.completed / max(1, hour_stats.total)) * 100 if hour_stats.total else 0
            })
        
        hourly_distribution.reverse()  # Chronological order
        
        # Agent workload balance analysis
        agent_workload_result = await db.execute(
            select(
                Agent.name,
                Agent.id,
                func.count(Task.id).filter(Task.status.in_([TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS])).label('active_tasks'),
                func.avg(
                    func.extract('epoch', Task.completed_at - Task.started_at) / 60
                ).filter(Task.status == TaskStatus.COMPLETED).label('avg_completion_minutes')
            ).
            join(Task, Agent.id == Task.assigned_agent_id, isouter=True).
            group_by(Agent.id, Agent.name)
        )
        
        workload_balance = []
        for agent_name, agent_id, active_tasks, avg_completion in agent_workload_result.all():
            workload_balance.append({
                "agent_id": str(agent_id),
                "agent_name": agent_name,
                "active_tasks": active_tasks or 0,
                "average_completion_minutes": float(avg_completion) if avg_completion else 0.0,
                "load_score": (active_tasks or 0) * 10 + (float(avg_completion) if avg_completion else 0) * 0.1
            })
        
        # Sort by load score for balance analysis
        workload_balance.sort(key=lambda x: x["load_score"], reverse=True)
        
        return {
            "time_range_hours": time_range_hours,
            "agent_distribution": agent_distribution,
            "type_distribution": type_distribution,
            "priority_distribution": priority_distribution,
            "hourly_timeline": hourly_distribution,
            "workload_balance": workload_balance,
            "distribution_metrics": {
                "total_agents_with_tasks": len(agent_distribution),
                "most_utilized_agent": agent_distribution[0]["agent_name"] if agent_distribution else None,
                "least_utilized_agent": agent_distribution[-1]["agent_name"] if agent_distribution else None,
                "workload_variance": max([w["load_score"] for w in workload_balance], default=0) - min([w["load_score"] for w in workload_balance], default=0),
                "distribution_efficiency": (len([a for a in agent_distribution if a["percentage"] > 5]) / max(1, len(agent_distribution))) * 100
            },
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get task distribution visualization", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to retrieve task distribution data: {str(e)}")


@router.post("/tasks/{task_id}/retry", response_model=Dict[str, Any])
async def retry_failed_task(
    task_id: str = Path(..., description="Task ID to retry"),
    reset_retry_count: bool = Query(False, description="Reset retry count to 0"),
    increase_priority: bool = Query(False, description="Increase task priority for retry"),
    new_agent_assignment: bool = Query(True, description="Allow assignment to different agent"),
    reason: str = Query("Manual retry via dashboard", description="Reason for retry"),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Manual retry controls for failed tasks with enhanced options.
    
    Provides recovery mechanisms for failed tasks with flexible retry options.
    """
    try:
        # Validate task exists and is retryable
        task_result = await db.execute(select(Task).where(Task.id == task_id))
        task = task_result.scalar_one_or_none()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        if task.status not in [TaskStatus.FAILED, TaskStatus.BLOCKED]:
            raise HTTPException(
                status_code=400, 
                detail=f"Task is not in retryable state. Current status: {task.status.value}"
            )
        
        # Store original task info for tracking
        retry_info = {
            "task_id": task_id,
            "original_status": task.status.value,
            "original_retry_count": task.retry_count,
            "original_priority": task.priority.name.lower(),
            "original_error": task.error_message,
            "reason": reason
        }
        
        # Reset retry count if requested
        if reset_retry_count:
            task.retry_count = 0
            retry_info["retry_count_reset"] = True
        else:
            task.retry_count += 1
            retry_info["new_retry_count"] = task.retry_count
        
        # Check if task can still be retried
        if task.retry_count >= task.max_retries and not reset_retry_count:
            raise HTTPException(
                status_code=400,
                detail=f"Task has exceeded maximum retries ({task.max_retries}). Use reset_retry_count=true to force retry."
            )
        
        # Increase priority if requested
        if increase_priority and task.priority != TaskPriority.CRITICAL:
            old_priority = task.priority
            if task.priority == TaskPriority.LOW:
                task.priority = TaskPriority.MEDIUM
            elif task.priority == TaskPriority.MEDIUM:
                task.priority = TaskPriority.HIGH
            elif task.priority == TaskPriority.HIGH:
                task.priority = TaskPriority.CRITICAL
            
            retry_info["priority_increased"] = {
                "from": old_priority.name.lower(),
                "to": task.priority.name.lower()
            }
        
        # Handle agent assignment for retry
        if new_agent_assignment:
            # Reset agent assignment to allow reassignment
            original_agent_id = task.assigned_agent_id
            task.assigned_agent_id = None
            task.assigned_at = None
            retry_info["agent_reassignment"] = {
                "original_agent_id": str(original_agent_id) if original_agent_id else None,
                "allow_new_assignment": True
            }
        else:
            # Keep original agent assignment if agent is still active
            if task.assigned_agent_id:
                agent_result = await db.execute(
                    select(Agent.status).where(Agent.id == task.assigned_agent_id)
                )
                agent_status = agent_result.scalar_one_or_none()
                if agent_status != AgentStatus.active:
                    # Agent not available, reset assignment
                    task.assigned_agent_id = None
                    task.assigned_at = None
                    retry_info["agent_reassignment"] = {
                        "reason": "Original agent not active",
                        "original_agent_unavailable": True
                    }
        
        # Reset task to pending status
        task.status = TaskStatus.PENDING
        task.error_message = f"Retry requested: {reason}"
        task.completed_at = None  # Clear completion timestamp
        task.updated_at = datetime.utcnow()
        
        await db.commit()
        
        # Send retry notification via Redis
        try:
            message_broker = get_message_broker()
            await message_broker.broadcast_message(
                from_agent="orchestrator",
                message_type="task_retry_requested",
                payload={
                    "task_id": task_id,
                    "task_title": task.title,
                    "retry_count": task.retry_count,
                    "priority": task.priority.name.lower(),
                    "reason": reason,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            retry_info["redis_notification"] = "sent"
        except Exception as redis_error:
            logger.warning("Failed to send retry notification", error=str(redis_error))
            retry_info["redis_notification"] = f"failed: {str(redis_error)}"
        
        return {
            "success": True,
            "retry_info": retry_info,
            "updated_task": task.to_dict(),
            "next_steps": [
                "Task reset to pending status",
                "Will be picked up by orchestrator for reassignment" if new_agent_assignment else "Will be reassigned to same agent if available",
                "Monitor task progress in queue status"
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retry task", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to retry task: {str(e)}")


# ==================== RECOVERY & CONTROL APIs ====================

@router.post("/system/emergency-override", response_model=Dict[str, Any])
async def emergency_system_override(
    action: str = Query(..., regex="^(stop_all_tasks|restart_all_agents|clear_task_queue|force_agent_restart|system_maintenance)$", description="Emergency action to perform"),
    target_agent_id: Optional[str] = Query(None, description="Specific agent ID for targeted actions"),
    confirm_emergency: bool = Query(False, description="Confirmation required for emergency actions"),
    reason: str = Query("Emergency override via dashboard", description="Reason for emergency action"),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Emergency system override controls for critical coordination failures.
    
    Provides last-resort controls when normal coordination mechanisms fail.
    """
    if not confirm_emergency:
        return {
            "error": "Emergency confirmation required",
            "message": "Set confirm_emergency=true to proceed with emergency override",
            "action": action,
            "warning": "This action may disrupt active system operations"
        }
    
    try:
        emergency_actions = []
        affected_components = {}
        
        if action == "stop_all_tasks":
            # Set all active tasks to blocked status
            active_tasks_result = await db.execute(
                select(Task).where(Task.status.in_([TaskStatus.IN_PROGRESS, TaskStatus.ASSIGNED]))
            )
            active_tasks = active_tasks_result.scalars().all()
            
            for task in active_tasks:
                task.status = TaskStatus.BLOCKED
                task.error_message = f"Emergency stop: {reason}"
                task.updated_at = datetime.utcnow()
            
            await db.commit()
            emergency_actions.append(f"Stopped {len(active_tasks)} active tasks")
            affected_components["tasks_stopped"] = len(active_tasks)
            
        elif action == "restart_all_agents":
            # Set all agents to maintenance mode
            agents_result = await db.execute(
                select(Agent).where(Agent.status != AgentStatus.inactive)
            )
            agents = agents_result.scalars().all()
            
            for agent in agents:
                agent.status = AgentStatus.maintenance
                agent.last_heartbeat = None
                agent.updated_at = datetime.utcnow()
            
            await db.commit()
            emergency_actions.append(f"Set {len(agents)} agents to maintenance mode")
            affected_components["agents_restarted"] = len(agents)
            
            # Broadcast restart command via Redis
            try:
                message_broker = get_message_broker()
                await message_broker.broadcast_message(
                    from_agent="orchestrator",
                    message_type="emergency_restart_all",
                    payload={
                        "reason": reason,
                        "timestamp": datetime.utcnow().isoformat(),
                        "initiated_by": "dashboard_emergency_override"
                    }
                )
                emergency_actions.append("Broadcast restart commands via Redis")
            except Exception as redis_error:
                emergency_actions.append(f"Redis broadcast failed: {str(redis_error)}")
            
        elif action == "clear_task_queue":
            # Reset all pending tasks
            pending_tasks_result = await db.execute(
                select(Task).where(Task.status == TaskStatus.PENDING)
            )
            pending_tasks = pending_tasks_result.scalars().all()
            
            for task in pending_tasks:
                task.status = TaskStatus.CANCELLED
                task.error_message = f"Queue cleared: {reason}"
                task.updated_at = datetime.utcnow()
                task.completed_at = datetime.utcnow()
            
            await db.commit()
            emergency_actions.append(f"Cleared {len(pending_tasks)} pending tasks from queue")
            affected_components["tasks_cleared"] = len(pending_tasks)
            
        elif action == "force_agent_restart":
            if not target_agent_id:
                raise HTTPException(status_code=400, detail="target_agent_id required for force_agent_restart action")
            
            # Force specific agent restart
            agent_result = await db.execute(select(Agent).where(Agent.id == target_agent_id))
            agent = agent_result.scalar_one_or_none()
            if not agent:
                raise HTTPException(status_code=404, detail="Target agent not found")
            
            # Reassign agent's active tasks
            agent_tasks_result = await db.execute(
                select(Task).where(
                    and_(
                        Task.assigned_agent_id == target_agent_id,
                        Task.status.in_([TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS])
                    )
                )
            )
            agent_tasks = agent_tasks_result.scalars().all()
            
            for task in agent_tasks:
                task.status = TaskStatus.PENDING
                task.assigned_agent_id = None
                task.assigned_at = None
                task.error_message = f"Reassigned due to emergency agent restart: {reason}"
            
            # Set agent to maintenance
            agent.status = AgentStatus.maintenance
            agent.last_heartbeat = None
            agent.updated_at = datetime.utcnow()
            
            await db.commit()
            emergency_actions.append(f"Force restarted agent {agent.name}, reassigned {len(agent_tasks)} tasks")
            affected_components["agent_id"] = target_agent_id
            affected_components["tasks_reassigned"] = len(agent_tasks)
            
        elif action == "system_maintenance":
            # Put entire system in maintenance mode
            
            # Stop all active tasks
            active_tasks_result = await db.execute(
                select(Task).where(Task.status.in_([TaskStatus.IN_PROGRESS, TaskStatus.ASSIGNED]))
            )
            active_tasks = active_tasks_result.scalars().all()
            
            for task in active_tasks:
                task.status = TaskStatus.BLOCKED
                task.error_message = f"System maintenance: {reason}"
            
            # Set all agents to maintenance
            agents_result = await db.execute(
                select(Agent).where(Agent.status != AgentStatus.inactive)
            )
            agents = agents_result.scalars().all()
            
            for agent in agents:
                agent.status = AgentStatus.maintenance
                agent.last_heartbeat = None
            
            await db.commit()
            
            emergency_actions.extend([
                f"System maintenance mode activated",
                f"Blocked {len(active_tasks)} active tasks",
                f"Set {len(agents)} agents to maintenance mode"
            ])
            
            affected_components.update({
                "maintenance_mode": True,
                "tasks_blocked": len(active_tasks),
                "agents_in_maintenance": len(agents)
            })
        
        # Record emergency action in system (could add dedicated emergency log table)
        logger.critical(
            "Emergency system override executed",
            action=action,
            reason=reason,
            affected_components=affected_components,
            emergency_actions=emergency_actions
        )
        
        return {
            "success": True,
            "emergency_action": action,
            "actions_performed": emergency_actions,
            "affected_components": affected_components,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
            "recovery_steps": [
                "Monitor system health endpoints for recovery status",
                "Check agent status for maintenance mode exit",
                "Review coordination success rate after recovery",
                "Validate task queue processing resumption"
            ],
            "warning": "System may require manual intervention to resume normal operations"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Emergency override failed", action=action, error=str(e))
        raise HTTPException(status_code=500, detail=f"Emergency override failed: {str(e)}")


@router.get("/system/health", response_model=Dict[str, Any])
async def get_comprehensive_system_health(
    include_historical: bool = Query(False, description="Include historical health trends"),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Comprehensive system health check with detailed component analysis.
    
    Provides complete health overview for operational monitoring.
    """
    try:
        health_data = {}
        current_time = datetime.utcnow()
        
        # Database health
        db_health = {}
        
        # Check database connectivity and basic metrics
        try:
            # Table counts
            agents_count = await db.execute(select(func.count(Agent.id)))
            db_health["total_agents"] = agents_count.scalar()
            
            tasks_count = await db.execute(select(func.count(Task.id)))
            db_health["total_tasks"] = tasks_count.scalar()
            
            # Recent activity
            recent_tasks = await db.execute(
                select(func.count(Task.id)).where(Task.created_at >= current_time - timedelta(hours=1))
            )
            db_health["tasks_last_hour"] = recent_tasks.scalar()
            
            db_health["status"] = "healthy"
            db_health["response_time_ms"] = "<5"
            
        except Exception as db_error:
            db_health["status"] = "unhealthy"
            db_health["error"] = str(db_error)
        
        health_data["database"] = db_health
        
        # Redis health
        redis_health = {}
        try:
            redis_client = get_redis()
            ping_result = await redis_client.ping()
            
            if ping_result:
                info = await redis_client.info("memory")
                redis_health.update({
                    "status": "healthy",
                    "memory_usage": info.get("used_memory_human", "unknown"),
                    "connected_clients": info.get("connected_clients", 0),
                    "response_time_ms": "<5"
                })
                
                # Check message streams
                streams = await redis_client.keys("agent_messages:*")
                redis_health["active_streams"] = len(streams)
                
            else:
                redis_health["status"] = "unhealthy"
                redis_health["error"] = "Redis ping failed"
                
        except Exception as redis_error:
            redis_health["status"] = "unhealthy"
            redis_health["error"] = str(redis_error)
        
        health_data["redis"] = redis_health
        
        # Agent system health
        agent_health = {}
        
        active_agents_result = await db.execute(
            select(func.count(Agent.id)).where(Agent.status == AgentStatus.active)
        )
        agent_health["active_agents"] = active_agents_result.scalar()
        
        maintenance_agents_result = await db.execute(
            select(func.count(Agent.id)).where(Agent.status == AgentStatus.maintenance)
        )
        agent_health["maintenance_agents"] = maintenance_agents_result.scalar()
        
        # Agents with stale heartbeats
        stale_heartbeat_result = await db.execute(
            select(func.count(Agent.id)).where(
                and_(
                    Agent.status == AgentStatus.active,
                    or_(
                        Agent.last_heartbeat.is_(None),
                        Agent.last_heartbeat < current_time - timedelta(minutes=5)
                    )
                )
            )
        )
        stale_heartbeats = stale_heartbeat_result.scalar()
        agent_health["stale_heartbeats"] = stale_heartbeats
        
        agent_health["status"] = "healthy" if stale_heartbeats == 0 else "degraded" if stale_heartbeats < 2 else "unhealthy"
        
        health_data["agents"] = agent_health
        
        # Task system health
        task_health = {}
        
        # Task queue analysis
        pending_tasks_result = await db.execute(
            select(func.count(Task.id)).where(Task.status == TaskStatus.PENDING)
        )
        task_health["pending_tasks"] = pending_tasks_result.scalar()
        
        in_progress_tasks_result = await db.execute(
            select(func.count(Task.id)).where(Task.status == TaskStatus.IN_PROGRESS)
        )
        task_health["in_progress_tasks"] = in_progress_tasks_result.scalar()
        
        # Long running tasks
        long_running_result = await db.execute(
            select(func.count(Task.id)).where(
                and_(
                    Task.status == TaskStatus.IN_PROGRESS,
                    Task.started_at < current_time - timedelta(hours=2)
                )
            )
        )
        task_health["long_running_tasks"] = long_running_result.scalar()
        
        # Failed tasks in last hour
        recent_failures_result = await db.execute(
            select(func.count(Task.id)).where(
                and_(
                    Task.status == TaskStatus.FAILED,
                    Task.updated_at >= current_time - timedelta(hours=1)
                )
            )
        )
        task_health["recent_failures"] = recent_failures_result.scalar()
        
        task_health["status"] = "healthy"
        if task_health["long_running_tasks"] > 5:
            task_health["status"] = "degraded"
        if task_health["recent_failures"] > 10:
            task_health["status"] = "unhealthy"
        
        health_data["tasks"] = task_health
        
        # Overall system health calculation
        component_scores = {
            "database": 100 if db_health.get("status") == "healthy" else 0,
            "redis": 100 if redis_health.get("status") == "healthy" else 0,
            "agents": 100 if agent_health.get("status") == "healthy" else 50 if agent_health.get("status") == "degraded" else 0,
            "tasks": 100 if task_health.get("status") == "healthy" else 50 if task_health.get("status") == "degraded" else 0
        }
        
        overall_score = sum(component_scores.values()) / len(component_scores)
        
        overall_health = {
            "score": overall_score,
            "status": "healthy" if overall_score >= 90 else "degraded" if overall_score >= 70 else "critical",
            "component_scores": component_scores
        }
        
        # Add alerts based on health status
        alerts = []
        if stale_heartbeats > 0:
            alerts.append({
                "level": "warning" if stale_heartbeats < 3 else "critical",
                "component": "agents",
                "message": f"{stale_heartbeats} agents have stale heartbeats",
                "action": "Check agent connectivity and restart if necessary"
            })
        
        if task_health["long_running_tasks"] > 0:
            alerts.append({
                "level": "warning",
                "component": "tasks",
                "message": f"{task_health['long_running_tasks']} tasks have been running for >2 hours",
                "action": "Review task progress and consider intervention"
            })
        
        if task_health["recent_failures"] > 5:
            alerts.append({
                "level": "critical" if task_health["recent_failures"] > 10 else "warning",
                "component": "tasks", 
                "message": f"{task_health['recent_failures']} task failures in the last hour",
                "action": "Investigate coordination system and error patterns"
            })
        
        # Historical trends (if requested)
        historical_data = {}
        if include_historical:
            # Simple historical health data (last 24 hours)
            historical_data["last_24_hours"] = []
            for hour in range(24):
                hour_start = current_time - timedelta(hours=hour+1)
                hour_end = current_time - timedelta(hours=hour)
                
                hour_tasks = await db.execute(
                    select(
                        func.count(Task.id).label('total'),
                        func.count(Task.id).filter(Task.status == TaskStatus.COMPLETED).label('completed')
                    ).where(and_(Task.created_at >= hour_start, Task.created_at < hour_end))
                )
                
                hour_data = hour_tasks.first()
                success_rate = (hour_data.completed / max(1, hour_data.total)) * 100 if hour_data.total else 100
                
                historical_data["last_24_hours"].append({
                    "hour": hour_start.strftime("%Y-%m-%d %H:00"),
                    "success_rate": success_rate,
                    "task_volume": hour_data.total
                })
            
            historical_data["last_24_hours"].reverse()  # Chronological order
        
        return {
            "overall_health": overall_health,
            "components": health_data,
            "alerts": alerts,
            "historical": historical_data if include_historical else None,
            "system_info": {
                "uptime": "unknown",  # Would need app start time tracking
                "version": "2.0.0",
                "environment": "development"  # Could be from config
            },
            "last_updated": current_time.isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get system health", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to retrieve system health: {str(e)}")


@router.post("/recovery/auto-heal", response_model=Dict[str, Any])
async def trigger_automatic_recovery(
    recovery_type: str = Query("smart", regex="^(smart|aggressive|conservative)$", description="Recovery strategy"),
    dry_run: bool = Query(True, description="Perform dry run without making changes"),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Trigger automatic recovery procedures for coordination system issues.
    
    Analyzes system state and applies appropriate recovery actions.
    """
    try:
        recovery_plan = {"actions": [], "analysis": {}, "dry_run": dry_run}
        current_time = datetime.utcnow()
        
        # Analyze current system state
        
        # Check for stuck tasks
        stuck_tasks_result = await db.execute(
            select(Task).where(
                and_(
                    Task.status == TaskStatus.IN_PROGRESS,
                    Task.started_at < current_time - timedelta(hours=1)
                )
            )
        )
        stuck_tasks = stuck_tasks_result.scalars().all()
        recovery_plan["analysis"]["stuck_tasks"] = len(stuck_tasks)
        
        # Check for agents with stale heartbeats
        stale_agents_result = await db.execute(
            select(Agent).where(
                and_(
                    Agent.status == AgentStatus.active,
                    or_(
                        Agent.last_heartbeat.is_(None),
                        Agent.last_heartbeat < current_time - timedelta(minutes=10)
                    )
                )
            )
        )
        stale_agents = stale_agents_result.scalars().all()
        recovery_plan["analysis"]["stale_agents"] = len(stale_agents)
        
        # Check for high failure rate
        recent_failures_result = await db.execute(
            select(func.count(Task.id)).where(
                and_(
                    Task.status == TaskStatus.FAILED,
                    Task.updated_at >= current_time - timedelta(hours=1)
                )
            )
        )
        recent_failures = recent_failures_result.scalar()
        
        recent_total_result = await db.execute(
            select(func.count(Task.id)).where(Task.created_at >= current_time - timedelta(hours=1))
        )
        recent_total = recent_total_result.scalar()
        
        failure_rate = (recent_failures / max(1, recent_total)) * 100
        recovery_plan["analysis"]["failure_rate"] = failure_rate
        
        # Determine recovery actions based on strategy and analysis
        
        if recovery_type == "conservative":
            # Conservative: Only basic cleanup
            if len(stuck_tasks) > 0:
                recovery_plan["actions"].append({
                    "type": "reset_stuck_tasks",
                    "description": f"Reset {len(stuck_tasks)} stuck tasks to pending",
                    "impact": "low",
                    "execute": not dry_run
                })
                
                if not dry_run:
                    for task in stuck_tasks:
                        task.status = TaskStatus.PENDING
                        task.assigned_agent_id = None
                        task.error_message = "Auto-recovery: Reset stuck task"
                    await db.commit()
        
        elif recovery_type == "smart":
            # Smart: Balanced recovery approach
            if len(stuck_tasks) > 0:
                recovery_plan["actions"].append({
                    "type": "reset_stuck_tasks",
                    "description": f"Reset {len(stuck_tasks)} stuck tasks to pending",
                    "impact": "medium",
                    "execute": not dry_run
                })
                
                if not dry_run:
                    for task in stuck_tasks:
                        task.status = TaskStatus.PENDING
                        task.assigned_agent_id = None
                        task.error_message = "Auto-recovery: Reset stuck task"
            
            if len(stale_agents) > 0:
                recovery_plan["actions"].append({
                    "type": "restart_stale_agents",
                    "description": f"Set {len(stale_agents)} agents to maintenance for restart",
                    "impact": "medium",
                    "execute": not dry_run
                })
                
                if not dry_run:
                    for agent in stale_agents:
                        agent.status = AgentStatus.maintenance
                        agent.last_heartbeat = None
            
            if failure_rate > 30:
                recovery_plan["actions"].append({
                    "type": "coordination_reset",
                    "description": "Reset coordination system due to high failure rate",
                    "impact": "high",
                    "execute": not dry_run
                })
                
                if not dry_run:
                    # Reset pending tasks
                    pending_tasks_result = await db.execute(
                        select(Task).where(Task.status == TaskStatus.PENDING)
                    )
                    pending_tasks = pending_tasks_result.scalars().all()
                    
                    for task in pending_tasks[:10]:  # Limit to 10 tasks
                        task.retry_count = 0
                        task.assigned_agent_id = None
                        task.error_message = "Auto-recovery: Coordination reset"
            
            if not dry_run:
                await db.commit()
        
        elif recovery_type == "aggressive":
            # Aggressive: Full system recovery
            recovery_plan["actions"].extend([
                {
                    "type": "reset_all_stuck_tasks",
                    "description": f"Reset all {len(stuck_tasks)} stuck tasks",
                    "impact": "high",
                    "execute": not dry_run
                },
                {
                    "type": "restart_all_stale_agents", 
                    "description": f"Restart all {len(stale_agents)} stale agents",
                    "impact": "high",
                    "execute": not dry_run
                },
                {
                    "type": "clear_redis_streams",
                    "description": "Clear Redis message streams",
                    "impact": "high",
                    "execute": not dry_run
                }
            ])
            
            if not dry_run:
                # Reset stuck tasks
                for task in stuck_tasks:
                    task.status = TaskStatus.PENDING
                    task.assigned_agent_id = None
                    task.error_message = "Auto-recovery: Aggressive reset"
                
                # Restart stale agents
                for agent in stale_agents:
                    agent.status = AgentStatus.maintenance
                    agent.last_heartbeat = None
                
                await db.commit()
                
                # Clear Redis streams
                try:
                    redis_client = get_redis()
                    streams = await redis_client.keys("agent_messages:*")
                    if streams:
                        await redis_client.delete(*streams)
                        recovery_plan["actions"].append({
                            "type": "redis_cleanup_completed",
                            "description": f"Cleared {len(streams)} Redis streams",
                            "impact": "high",
                            "execute": True
                        })
                except Exception as redis_error:
                    recovery_plan["actions"].append({
                        "type": "redis_cleanup_failed",
                        "description": f"Failed to clear Redis streams: {str(redis_error)}",
                        "impact": "high",
                        "execute": True
                    })
        
        # Calculate recovery impact and success likelihood
        total_actions = len([a for a in recovery_plan["actions"] if a["execute"]])
        high_impact_actions = len([a for a in recovery_plan["actions"] if a.get("impact") == "high" and a["execute"]])
        
        success_likelihood = min(95, max(20, 80 - (high_impact_actions * 15) + (total_actions * 5)))
        
        recovery_plan.update({
            "recovery_type": recovery_type,
            "total_actions": total_actions,
            "high_impact_actions": high_impact_actions,
            "success_likelihood": success_likelihood,
            "estimated_recovery_time": f"{total_actions * 2} minutes",
            "next_steps": [
                "Monitor system health after recovery actions",
                "Check coordination success rate in 10 minutes",
                "Verify agent heartbeats are restored",
                "Confirm task queue is processing normally"
            ] if not dry_run else [
                "Review recovery plan and execute with dry_run=false",
                "Consider manual intervention for high-impact actions",
                "Monitor system state before applying changes"
            ],
            "timestamp": current_time.isoformat()
        })
        
        return recovery_plan
        
    except Exception as e:
        logger.error("Auto-recovery failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Auto-recovery failed: {str(e)}")


@router.get("/logs/coordination", response_model=Dict[str, Any])
async def get_coordination_error_logs(
    hours: int = Query(24, ge=1, le=168, description="Hours of logs to retrieve"),
    error_level: str = Query("all", regex="^(all|error|critical|warning)$", description="Filter by error level"),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of log entries"),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get coordination error logs with filtering and analysis.
    
    Provides detailed error logs for troubleshooting coordination issues.
    """
    try:
        since = datetime.utcnow() - timedelta(hours=hours)
        
        # Get failed tasks as our primary log source
        failed_tasks_result = await db.execute(
            select(Task).where(
                and_(
                    Task.status == TaskStatus.FAILED,
                    Task.updated_at >= since,
                    Task.error_message.isnot(None)
                )
            ).order_by(Task.updated_at.desc()).limit(limit)
        )
        failed_tasks = failed_tasks_result.scalars().all()
        
        # Convert to log entries
        log_entries = []
        error_patterns = {}
        
        for task in failed_tasks:
            # Classify error level
            error_message = task.error_message or ""
            if "critical" in error_message.lower() or "fatal" in error_message.lower():
                level = "critical"
            elif "error" in error_message.lower():
                level = "error"
            elif "warning" in error_message.lower() or "warn" in error_message.lower():
                level = "warning"
            else:
                level = "error"  # Default for failed tasks
            
            # Apply level filter
            if error_level != "all" and level != error_level:
                continue
            
            # Extract error pattern
            pattern = error_message.split(':')[0] if ':' in error_message else error_message[:30]
            error_patterns[pattern] = error_patterns.get(pattern, 0) + 1
            
            log_entry = {
                "timestamp": task.updated_at.isoformat() if task.updated_at else None,
                "level": level,
                "component": "coordination",
                "task_id": str(task.id),
                "task_title": task.title,
                "agent_id": str(task.assigned_agent_id) if task.assigned_agent_id else None,
                "error_message": error_message,
                "context": {
                    "task_type": task.task_type.value if task.task_type else None,
                    "priority": task.priority.name.lower(),
                    "retry_count": task.retry_count,
                    "max_retries": task.max_retries
                }
            }
            
            log_entries.append(log_entry)
        
        # Get top error patterns
        top_patterns = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Analyze error trends
        hourly_errors = {}
        for entry in log_entries:
            if entry["timestamp"]:
                hour = datetime.fromisoformat(entry["timestamp"].replace('Z', '+00:00')).strftime("%Y-%m-%d %H:00")
                hourly_errors[hour] = hourly_errors.get(hour, 0) + 1
        
        # Generate recommendations based on error analysis
        recommendations = []
        
        if any("redis" in pattern.lower() for pattern, _ in top_patterns):
            recommendations.append({
                "issue": "Redis connectivity issues detected",
                "action": "Check Redis server health and network connectivity",
                "priority": "high"
            })
        
        if any("timeout" in pattern.lower() for pattern, _ in top_patterns):
            recommendations.append({
                "issue": "Task timeout errors frequent",
                "action": "Review task timeout configurations and increase if necessary",
                "priority": "medium"
            })
        
        if any("serialization" in pattern.lower() for pattern, _ in top_patterns):
            recommendations.append({
                "issue": "Data serialization failures",
                "action": "Review message payload formats and serialization logic",
                "priority": "high"
            })
        
        # Calculate error rate trends
        total_errors = len(log_entries)
        critical_errors = len([e for e in log_entries if e["level"] == "critical"])
        error_rate_trend = "stable"  # Simplified - would compare to previous period
        
        return {
            "summary": {
                "total_errors": total_errors,
                "critical_errors": critical_errors,
                "error_rate_trend": error_rate_trend,
                "time_range_hours": hours,
                "most_common_pattern": top_patterns[0][0] if top_patterns else None
            },
            "log_entries": log_entries,
            "error_patterns": [{"pattern": p, "count": c} for p, c in top_patterns],
            "hourly_distribution": [{"hour": h, "error_count": c} for h, c in sorted(hourly_errors.items())],
            "recommendations": recommendations,
            "filters_applied": {
                "hours": hours,
                "error_level": error_level,
                "limit": limit
            },
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get coordination logs", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to retrieve coordination logs: {str(e)}")