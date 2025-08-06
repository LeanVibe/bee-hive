"""
Redis-based Shared World State for LeanVibe Agent Hive 2.0

Provides centralized, distributed state management for multi-agent coordination
with atomic operations, high performance, and fault tolerance.
"""

import asyncio
import json
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
import structlog

from .redis import get_redis
from contextlib import asynccontextmanager

logger = structlog.get_logger(__name__)


class TaskStatus(Enum):
    """Enumeration for task statuses."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class SharedWorldState:
    """
    Redis-based shared state management for multi-agent coordination.
    
    Provides atomic operations for task status, agent load tracking,
    and workflow progress in a distributed environment.
    """

    def __init__(self, redis_prefix: str = "world_state"):
        """Initialize SharedWorldState with Redis backend."""
        self.redis_prefix = redis_prefix
        self.tasks_key = f"{redis_prefix}:tasks"
        self.agents_key = f"{redis_prefix}:agents"
        self.workflows_key = f"{redis_prefix}:workflows"
        
    @asynccontextmanager
    async def _get_redis(self):
        """Get Redis connection with proper error handling."""
        redis_client = get_redis()
        try:
            yield redis_client
        except Exception as e:
            logger.error("Redis operation failed", error=str(e))
            raise

    # --- Task Management ---

    async def add_task(self, task_id: str, task_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a new task to the shared state with 'pending' status.
        
        Args:
            task_id: Unique identifier for the task
            task_data: Optional additional task information
            
        Raises:
            ValueError: If task already exists
        """
        async with self._get_redis() as redis:
            task_key = f"{self.tasks_key}:{task_id}"
            
            # Check if task already exists
            if await redis.exists(task_key):
                raise ValueError(f"Task with ID '{task_id}' already exists.")
            
            task_info = {
                "status": TaskStatus.PENDING.value,
                "data": task_data or {},
                "created_at": time.time(),
                "updated_at": time.time(),
                "agent_id": None,
            }
            
            await redis.hset(task_key, mapping={
                k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                for k, v in task_info.items()
            })
            
            logger.info("Task added to shared state", task_id=task_id, status=TaskStatus.PENDING.value)

    async def update_task_status(
        self, 
        task_id: str, 
        status: TaskStatus, 
        agent_id: Optional[str] = None
    ) -> None:
        """
        Update task status with atomic agent load adjustments.
        
        Args:
            task_id: ID of task to update
            status: New task status
            agent_id: Agent associated with status change
            
        Raises:
            KeyError: If task doesn't exist
            ValueError: If agent_id required but not provided
        """
        async with self._get_redis() as redis:
            task_key = f"{self.tasks_key}:{task_id}"
            
            # Start Redis transaction
            async with redis.pipeline() as pipe:
                while True:
                    try:
                        # Watch the task key for changes during transaction
                        await pipe.watch(task_key)
                        
                        # Get current task info
                        task_info = await redis.hgetall(task_key)
                        if not task_info:
                            raise KeyError(f"Task with ID '{task_id}' not found.")
                        
                        old_status = TaskStatus(task_info.get('status', TaskStatus.PENDING.value))
                        
                        if old_status == status:
                            await pipe.unwatch()
                            return  # No change needed
                        
                        # Start transaction
                        pipe.multi()
                        
                        # Handle agent load changes
                        if old_status == TaskStatus.IN_PROGRESS:
                            previous_agent_id = task_info.get('agent_id')
                            if previous_agent_id and previous_agent_id != 'None':
                                agent_key = f"{self.agents_key}:{previous_agent_id}"
                                await pipe.hincrby(agent_key, "load", -1)
                                await pipe.srem(f"{agent_key}:tasks", task_id)
                        
                        if status == TaskStatus.IN_PROGRESS:
                            if not agent_id:
                                raise ValueError("Agent ID required for IN_PROGRESS status")
                            
                            agent_key = f"{self.agents_key}:{agent_id}"
                            # Ensure agent exists
                            await pipe.hsetnx(agent_key, "registered_at", time.time())
                            await pipe.hincrby(agent_key, "load", 1)
                            await pipe.sadd(f"{agent_key}:tasks", task_id)
                        
                        # Update task
                        await pipe.hset(task_key, mapping={
                            "status": status.value,
                            "updated_at": str(time.time()),
                            "agent_id": agent_id or ""
                        })
                        
                        # Execute transaction
                        await pipe.execute()
                        
                        logger.info("Task status updated", 
                                  task_id=task_id, 
                                  old_status=old_status.value,
                                  new_status=status.value,
                                  agent_id=agent_id)
                        break
                        
                    except redis.WatchError:
                        # Task was modified during transaction, retry
                        logger.warning("Task modified during update, retrying", task_id=task_id)
                        continue

    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task information by ID."""
        async with self._get_redis() as redis:
            task_key = f"{self.tasks_key}:{task_id}"
            task_info = await redis.hgetall(task_key)
            
            if not task_info:
                return None
            
            # Parse JSON fields back to objects
            parsed_info = {}
            for k, v in task_info.items():
                if k == "data":
                    try:
                        parsed_info[k] = json.loads(v) if v else {}
                    except json.JSONDecodeError:
                        parsed_info[k] = {}
                else:
                    parsed_info[k] = v
            
            return parsed_info

    async def get_tasks_by_status(self, status: TaskStatus) -> List[Dict[str, Any]]:
        """Get all tasks with a specific status."""
        async with self._get_redis() as redis:
            # Scan for all task keys
            tasks = []
            async for key in redis.scan_iter(match=f"{self.tasks_key}:*"):
                task_info = await redis.hgetall(key)
                if task_info.get('status') == status.value:
                    task_id = key.split(':')[-1]
                    task_data = await self.get_task(task_id)
                    if task_data:
                        task_data['id'] = task_id
                        tasks.append(task_data)
            
            return tasks

    # --- Agent Load Monitoring ---

    async def register_agent(self, agent_id: str, agent_data: Optional[Dict[str, Any]] = None) -> None:
        """Register an agent in the shared state."""
        async with self._get_redis() as redis:
            agent_key = f"{self.agents_key}:{agent_id}"
            
            agent_info = {
                "load": "0",
                "registered_at": str(time.time()),
                "data": json.dumps(agent_data or {})
            }
            
            await redis.hset(agent_key, mapping=agent_info)
            logger.info("Agent registered", agent_id=agent_id)

    async def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent information including current load."""
        async with self._get_redis() as redis:
            agent_key = f"{self.agents_key}:{agent_id}"
            agent_info = await redis.hgetall(agent_key)
            
            if not agent_info:
                return None
            
            # Get agent's current tasks
            current_tasks = await redis.smembers(f"{agent_key}:tasks")
            
            return {
                "agent_id": agent_id,
                "load": int(agent_info.get("load", 0)),
                "registered_at": float(agent_info.get("registered_at", 0)),
                "data": json.loads(agent_info.get("data", "{}")),
                "current_tasks": list(current_tasks)
            }

    async def get_least_loaded_agent(self, exclude_agents: Optional[Set[str]] = None) -> Optional[str]:
        """Find the agent with the lowest current load."""
        async with self._get_redis() as redis:
            exclude_agents = exclude_agents or set()
            
            min_load = float('inf')
            best_agent = None
            
            async for key in redis.scan_iter(match=f"{self.agents_key}:*"):
                agent_id = key.split(':')[-1]
                if agent_id in exclude_agents:
                    continue
                
                load = int(await redis.hget(key, "load") or 0)
                if load < min_load:
                    min_load = load
                    best_agent = agent_id
            
            return best_agent

    # --- Workflow Progress Tracking ---

    async def create_workflow(self, workflow_id: str, task_ids: List[str]) -> None:
        """Create a workflow composed of multiple tasks."""
        async with self._get_redis() as redis:
            workflow_key = f"{self.workflows_key}:{workflow_id}"
            
            if await redis.exists(workflow_key):
                raise ValueError(f"Workflow with ID '{workflow_id}' already exists.")
            
            # Verify all tasks exist
            for task_id in task_ids:
                task_key = f"{self.tasks_key}:{task_id}"
                if not await redis.exists(task_key):
                    raise KeyError(f"Task with ID '{task_id}' not found.")
            
            workflow_info = {
                "created_at": str(time.time()),
                "task_count": str(len(task_ids))
            }
            
            async with redis.pipeline() as pipe:
                await pipe.hset(workflow_key, mapping=workflow_info)
                await pipe.sadd(f"{workflow_key}:tasks", *task_ids)
                await pipe.execute()
            
            logger.info("Workflow created", workflow_id=workflow_id, task_count=len(task_ids))

    async def get_workflow_progress(self, workflow_id: str) -> Dict[str, Any]:
        """Calculate workflow progress based on task statuses."""
        async with self._get_redis() as redis:
            workflow_key = f"{self.workflows_key}:{workflow_id}"
            
            if not await redis.exists(workflow_key):
                raise KeyError(f"Workflow with ID '{workflow_id}' not found.")
            
            # Get workflow task IDs
            task_ids = await redis.smembers(f"{workflow_key}:tasks")
            total_tasks = len(task_ids)
            
            if total_tasks == 0:
                return {
                    "workflow_id": workflow_id,
                    "total_tasks": 0,
                    "completed_tasks": 0,
                    "progress_percent": 100.0,
                    "status_breakdown": {}
                }
            
            # Count task statuses
            status_counts = {status.value: 0 for status in TaskStatus}
            
            for task_id in task_ids:
                task_status = await redis.hget(f"{self.tasks_key}:{task_id}", "status")
                if task_status:
                    status_counts[task_status] = status_counts.get(task_status, 0) + 1
            
            completed_count = status_counts.get(TaskStatus.COMPLETED.value, 0)
            progress = (completed_count / total_tasks) * 100 if total_tasks > 0 else 0
            
            return {
                "workflow_id": workflow_id,
                "total_tasks": total_tasks,
                "completed_tasks": completed_count,
                "progress_percent": round(progress, 2),
                "status_breakdown": status_counts
            }

    # --- System Health and Monitoring ---

    async def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system state overview."""
        async with self._get_redis() as redis:
            # Count tasks by status
            task_counts = {status.value: 0 for status in TaskStatus}
            total_tasks = 0
            
            async for key in redis.scan_iter(match=f"{self.tasks_key}:*"):
                total_tasks += 1
                task_status = await redis.hget(key, "status")
                if task_status:
                    task_counts[task_status] = task_counts.get(task_status, 0) + 1
            
            # Count agents and their loads
            agent_count = 0
            total_load = 0
            
            async for key in redis.scan_iter(match=f"{self.agents_key}:*"):
                if ":tasks" not in key:  # Skip task list keys
                    agent_count += 1
                    load = int(await redis.hget(key, "load") or 0)
                    total_load += load
            
            # Count workflows
            workflow_count = 0
            async for key in redis.scan_iter(match=f"{self.workflows_key}:*"):
                if ":tasks" not in key:  # Skip task list keys
                    workflow_count += 1
            
            return {
                "timestamp": time.time(),
                "tasks": {
                    "total": total_tasks,
                    "by_status": task_counts
                },
                "agents": {
                    "total": agent_count,
                    "total_load": total_load,
                    "average_load": round(total_load / agent_count, 2) if agent_count > 0 else 0
                },
                "workflows": {
                    "total": workflow_count
                }
            }

    async def cleanup_completed_tasks(self, older_than_hours: int = 24) -> int:
        """Clean up completed tasks older than specified hours."""
        async with self._get_redis() as redis:
            cutoff_time = time.time() - (older_than_hours * 3600)
            cleaned_count = 0
            
            async for key in redis.scan_iter(match=f"{self.tasks_key}:*"):
                task_info = await redis.hgetall(key)
                
                if (task_info.get('status') == TaskStatus.COMPLETED.value and
                    float(task_info.get('updated_at', 0)) < cutoff_time):
                    
                    await redis.delete(key)
                    cleaned_count += 1
            
            logger.info("Completed tasks cleaned up", count=cleaned_count, cutoff_hours=older_than_hours)
            return cleaned_count


# Global shared world state instance
_shared_world_state: Optional[SharedWorldState] = None


def get_shared_world_state() -> SharedWorldState:
    """Get the global shared world state instance."""
    global _shared_world_state
    if _shared_world_state is None:
        _shared_world_state = SharedWorldState()
    return _shared_world_state


async def initialize_shared_world_state() -> SharedWorldState:
    """Initialize and return the global shared world state."""
    return get_shared_world_state()