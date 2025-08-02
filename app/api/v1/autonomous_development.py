"""
Autonomous Development API for LeanVibe Agent Hive 2.0

REST API endpoints for managing autonomous development workflows:
- Task creation and management
- AI worker control and monitoring  
- Real-time progress tracking
- Results and analytics

Provides production-ready interface for autonomous development capabilities.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import uuid

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from pydantic import BaseModel, Field

from ...core.database import get_session
from ...core.ai_task_worker import create_ai_worker, stop_ai_worker, get_worker_stats
from ...core.ai_gateway import AIModel
from ...core.task_queue import TaskQueue
from ...models.task import Task, TaskStatus, TaskType, TaskPriority
from ...models.agent import Agent

router = APIRouter(prefix="/autonomous-development", tags=["Autonomous Development"])


# Request/Response Models

class CreateTaskRequest(BaseModel):
    """Request model for creating autonomous development tasks."""
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    task_type: TaskType
    priority: TaskPriority = TaskPriority.MEDIUM
    required_capabilities: List[str] = Field(default_factory=list)
    estimated_effort: Optional[int] = Field(None, ge=1, le=1440)  # 1 minute to 24 hours
    context: Dict[str, Any] = Field(default_factory=dict)
    due_date: Optional[datetime] = None


class TaskResponse(BaseModel):
    """Response model for task information."""
    id: str
    title: str
    description: Optional[str]
    task_type: Optional[str]
    status: str
    priority: str
    assigned_agent_id: Optional[str]
    created_by_agent_id: Optional[str]
    required_capabilities: List[str]
    estimated_effort: Optional[int]
    actual_effort: Optional[int]
    result: Dict[str, Any]
    error_message: Optional[str]
    retry_count: int
    created_at: datetime
    updated_at: datetime
    assigned_at: Optional[datetime]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    due_date: Optional[datetime]


class CreateWorkerRequest(BaseModel):
    """Request model for creating AI workers."""
    worker_id: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)
    ai_model: AIModel = AIModel.CLAUDE_3_5_SONNET


class WorkerResponse(BaseModel):
    """Response model for worker information."""
    worker_id: str
    status: str
    capabilities: List[str]
    ai_model: str
    current_task_id: Optional[str]
    uptime_seconds: float
    tasks_processed: int
    tasks_completed: int
    tasks_failed: int
    average_processing_time: float
    tasks_per_hour: float


class AutonomousProjectRequest(BaseModel):
    """Request model for autonomous project creation."""
    project_name: str = Field(..., min_length=1, max_length=100)
    project_type: str = Field(..., description="Type of project: web_api, frontend_app, data_pipeline, etc.")
    requirements: str = Field(..., min_length=10, description="Detailed project requirements")
    technology_stack: List[str] = Field(default_factory=list)
    priority: TaskPriority = TaskPriority.MEDIUM
    estimated_duration_hours: Optional[int] = Field(None, ge=1, le=168)  # 1 hour to 1 week


class ProjectResponse(BaseModel):
    """Response model for autonomous project."""
    project_id: str
    project_name: str
    project_type: str
    status: str
    task_count: int
    completed_tasks: int
    failed_tasks: int
    estimated_completion: Optional[datetime]
    created_at: datetime
    tasks: List[TaskResponse]


# API Endpoints

@router.post("/tasks", response_model=TaskResponse)
async def create_task(
    request: CreateTaskRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_session)
) -> TaskResponse:
    """Create a new autonomous development task."""
    
    # Create task
    task = Task(
        id=uuid.uuid4(),
        title=request.title,
        description=request.description,
        task_type=request.task_type,
        priority=request.priority,
        required_capabilities=request.required_capabilities,
        estimated_effort=request.estimated_effort,
        context=request.context,
        due_date=request.due_date
    )
    
    db.add(task)
    await db.commit()
    await db.refresh(task)
    
    # Enqueue task for processing
    background_tasks.add_task(_enqueue_task, task.id, request.priority, request.required_capabilities)
    
    return _task_to_response(task)


@router.get("/tasks", response_model=List[TaskResponse])
async def list_tasks(
    status: Optional[TaskStatus] = None,
    task_type: Optional[TaskType] = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_session)
) -> List[TaskResponse]:
    """List autonomous development tasks with filtering."""
    
    query = select(Task)
    
    if status:
        query = query.where(Task.status == status)
    
    if task_type:
        query = query.where(Task.task_type == task_type)
    
    query = query.order_by(Task.created_at.desc()).offset(offset).limit(limit)
    
    result = await db.execute(query)
    tasks = result.scalars().all()
    
    return [_task_to_response(task) for task in tasks]


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: str,
    db: AsyncSession = Depends(get_session)
) -> TaskResponse:
    """Get a specific task by ID."""
    
    try:
        task_uuid = uuid.UUID(task_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task ID format")
    
    result = await db.execute(select(Task).where(Task.id == task_uuid))
    task = result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return _task_to_response(task)


@router.delete("/tasks/{task_id}")
async def cancel_task(
    task_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_session)
) -> Dict[str, str]:
    """Cancel a pending or in-progress task."""
    
    try:
        task_uuid = uuid.UUID(task_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task ID format")
    
    result = await db.execute(select(Task).where(Task.id == task_uuid))
    task = result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
        raise HTTPException(status_code=400, detail="Cannot cancel completed task")
    
    # Update task status
    task.status = TaskStatus.CANCELLED
    task.error_message = "Cancelled by user request"
    await db.commit()
    
    # Cancel in queue
    background_tasks.add_task(_cancel_task_in_queue, task_uuid)
    
    return {"message": "Task cancelled successfully"}


@router.post("/workers", response_model=WorkerResponse)
async def create_worker(request: CreateWorkerRequest) -> WorkerResponse:
    """Create and start a new AI worker."""
    
    try:
        worker = await create_ai_worker(
            worker_id=request.worker_id,
            capabilities=request.capabilities or [
                "code_generation", "code_review", "testing", 
                "documentation", "architecture", "debugging"
            ],
            ai_model=request.ai_model
        )
        
        stats = await worker.get_stats()
        return _worker_stats_to_response(stats)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create worker: {str(e)}")


@router.get("/workers", response_model=List[WorkerResponse])
async def list_workers() -> List[WorkerResponse]:
    """List all active AI workers."""
    
    worker_stats = await get_worker_stats()
    
    return [
        _worker_stats_to_response(stats) 
        for stats in worker_stats.get("worker_details", {}).values()
    ]


@router.delete("/workers/{worker_id}")
async def stop_worker(worker_id: str) -> Dict[str, str]:
    """Stop a specific AI worker."""
    
    success = await stop_ai_worker(worker_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Worker not found")
    
    return {"message": "Worker stopped successfully"}


@router.get("/stats")
async def get_system_stats(db: AsyncSession = Depends(get_session)) -> Dict[str, Any]:
    """Get overall autonomous development system statistics."""
    
    # Task statistics
    task_stats = await db.execute(
        select(
            func.count(Task.id).label("total_tasks"),
            func.count().filter(Task.status == TaskStatus.PENDING).label("pending_tasks"),
            func.count().filter(Task.status == TaskStatus.IN_PROGRESS).label("in_progress_tasks"),
            func.count().filter(Task.status == TaskStatus.COMPLETED).label("completed_tasks"),
            func.count().filter(Task.status == TaskStatus.FAILED).label("failed_tasks"),
            func.avg(Task.actual_effort).label("avg_processing_time")
        )
    )
    
    task_row = task_stats.first()
    
    # Worker statistics
    worker_stats = await get_worker_stats()
    
    # Recent activity (last 24 hours)
    recent_cutoff = datetime.utcnow() - timedelta(hours=24)
    recent_activity = await db.execute(
        select(
            func.count().filter(Task.created_at >= recent_cutoff).label("tasks_created_24h"),
            func.count().filter(
                and_(Task.completed_at >= recent_cutoff, Task.status == TaskStatus.COMPLETED)
            ).label("tasks_completed_24h")
        )
    )
    
    recent_row = recent_activity.first()
    
    return {
        "system_status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "task_statistics": {
            "total_tasks": task_row.total_tasks or 0,
            "pending_tasks": task_row.pending_tasks or 0,
            "in_progress_tasks": task_row.in_progress_tasks or 0,
            "completed_tasks": task_row.completed_tasks or 0,
            "failed_tasks": task_row.failed_tasks or 0,
            "success_rate": (
                (task_row.completed_tasks or 0) / max(1, (task_row.completed_tasks or 0) + (task_row.failed_tasks or 0))
            ) * 100,
            "average_processing_time_minutes": task_row.avg_processing_time or 0
        },
        "worker_statistics": {
            "active_workers": worker_stats.get("active_workers", 0),
            "total_tasks_processed": worker_stats.get("total_tasks_processed", 0),
            "total_tasks_completed": worker_stats.get("total_tasks_completed", 0),
            "total_tasks_failed": worker_stats.get("total_tasks_failed", 0)
        },
        "recent_activity": {
            "tasks_created_last_24h": recent_row.tasks_created_24h or 0,
            "tasks_completed_last_24h": recent_row.tasks_completed_24h or 0,
            "throughput_per_hour": (recent_row.tasks_completed_24h or 0) / 24.0
        }
    }


@router.post("/projects", response_model=ProjectResponse)
async def create_autonomous_project(
    request: AutonomousProjectRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_session)
) -> ProjectResponse:
    """Create a complete autonomous development project with multiple coordinated tasks."""
    
    project_id = str(uuid.uuid4())
    
    # Define project templates
    project_tasks = _generate_project_tasks(request, project_id)
    
    task_responses = []
    
    for task_data in project_tasks:
        task = Task(
            id=uuid.uuid4(),
            title=task_data["title"],
            description=task_data["description"],
            task_type=task_data["task_type"],
            priority=request.priority,
            required_capabilities=task_data["required_capabilities"],
            estimated_effort=task_data["estimated_effort"],
            context={
                **task_data.get("context", {}),
                "project_id": project_id,
                "project_name": request.project_name,
                "project_type": request.project_type,
                "technology_stack": request.technology_stack
            }
        )
        
        db.add(task)
        task_responses.append(task)
        
        # Enqueue task
        background_tasks.add_task(
            _enqueue_task, 
            task.id, 
            request.priority, 
            task_data["required_capabilities"]
        )
    
    await db.commit()
    
    # Refresh all tasks
    for task in task_responses:
        await db.refresh(task)
    
    return ProjectResponse(
        project_id=project_id,
        project_name=request.project_name,
        project_type=request.project_type,
        status="in_progress",
        task_count=len(task_responses),
        completed_tasks=0,
        failed_tasks=0,
        estimated_completion=_calculate_estimated_completion(task_responses),
        created_at=datetime.utcnow(),
        tasks=[_task_to_response(task) for task in task_responses]
    )


@router.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    db: AsyncSession = Depends(get_session)
) -> ProjectResponse:
    """Get autonomous project status and progress."""
    
    # Get all tasks for this project
    result = await db.execute(
        select(Task).where(Task.context["project_id"].astext == project_id)
    )
    tasks = result.scalars().all()
    
    if not tasks:
        raise HTTPException(status_code=404, detail="Project not found")
    
    completed_tasks = sum(1 for task in tasks if task.status == TaskStatus.COMPLETED)
    failed_tasks = sum(1 for task in tasks if task.status == TaskStatus.FAILED)
    
    # Determine overall project status
    if completed_tasks == len(tasks):
        project_status = "completed"
    elif failed_tasks > 0:
        project_status = "partial_failure"
    elif any(task.status == TaskStatus.IN_PROGRESS for task in tasks):
        project_status = "in_progress"
    else:
        project_status = "pending"
    
    # Get project details from first task
    first_task = tasks[0]
    
    return ProjectResponse(
        project_id=project_id,
        project_name=first_task.context.get("project_name", "Unknown Project"),
        project_type=first_task.context.get("project_type", "unknown"),
        status=project_status,
        task_count=len(tasks),
        completed_tasks=completed_tasks,
        failed_tasks=failed_tasks,
        estimated_completion=_calculate_estimated_completion(tasks),
        created_at=min(task.created_at for task in tasks),
        tasks=[_task_to_response(task) for task in tasks]
    )


# Helper Functions

def _task_to_response(task: Task) -> TaskResponse:
    """Convert Task model to TaskResponse."""
    return TaskResponse(
        id=str(task.id),
        title=task.title,
        description=task.description,
        task_type=task.task_type.value if task.task_type else None,
        status=task.status.value,
        priority=task.priority.name.lower(),
        assigned_agent_id=str(task.assigned_agent_id) if task.assigned_agent_id else None,
        created_by_agent_id=str(task.created_by_agent_id) if task.created_by_agent_id else None,
        required_capabilities=task.required_capabilities or [],
        estimated_effort=task.estimated_effort,
        actual_effort=task.actual_effort,
        result=task.result or {},
        error_message=task.error_message,
        retry_count=task.retry_count,
        created_at=task.created_at,
        updated_at=task.updated_at,
        assigned_at=task.assigned_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        due_date=task.due_date
    )


def _worker_stats_to_response(stats: Dict[str, Any]) -> WorkerResponse:
    """Convert worker stats to WorkerResponse."""
    return WorkerResponse(
        worker_id=stats["worker_id"],
        status=stats["status"],
        capabilities=stats["capabilities"],
        ai_model=stats["ai_model"],
        current_task_id=stats["current_task_id"],
        uptime_seconds=stats["uptime_seconds"],
        tasks_processed=stats["tasks_processed"],
        tasks_completed=stats["tasks_completed"],
        tasks_failed=stats["tasks_failed"],
        average_processing_time=stats["average_processing_time"],
        tasks_per_hour=stats["tasks_per_hour"]
    )


async def _enqueue_task(
    task_id: uuid.UUID, 
    priority: TaskPriority, 
    required_capabilities: List[str]
) -> None:
    """Background task to enqueue a task."""
    task_queue = TaskQueue()
    await task_queue.start()
    
    try:
        await task_queue.enqueue_task(
            task_id=task_id,
            priority=priority,
            required_capabilities=required_capabilities
        )
    finally:
        await task_queue.stop()


async def _cancel_task_in_queue(task_id: uuid.UUID) -> None:
    """Background task to cancel a task in the queue."""
    task_queue = TaskQueue()
    await task_queue.start()
    
    try:
        await task_queue.cancel_task(task_id)
    finally:
        await task_queue.stop()


def _generate_project_tasks(request: AutonomousProjectRequest, project_id: str) -> List[Dict[str, Any]]:
    """Generate tasks for different project types."""
    
    if request.project_type == "web_api":
        return [
            {
                "title": f"Design {request.project_name} API Architecture",
                "description": f"Design the overall architecture for {request.project_name} including data models, API endpoints, and service structure. Requirements: {request.requirements}",
                "task_type": TaskType.ARCHITECTURE,
                "required_capabilities": ["architecture", "api_development"],
                "estimated_effort": 60
            },
            {
                "title": f"Implement {request.project_name} Core API",
                "description": f"Implement the core API endpoints and business logic for {request.project_name}. Technology stack: {request.technology_stack}. Requirements: {request.requirements}",
                "task_type": TaskType.CODE_GENERATION,
                "required_capabilities": ["code_generation", "api_development"],
                "estimated_effort": 120
            },
            {
                "title": f"Create Tests for {request.project_name}",
                "description": f"Create comprehensive unit and integration tests for {request.project_name} with high coverage.",
                "task_type": TaskType.TESTING,
                "required_capabilities": ["testing", "code_review"],
                "estimated_effort": 90
            },
            {
                "title": f"Document {request.project_name} API",
                "description": f"Create comprehensive API documentation including setup guide, endpoint reference, and usage examples for {request.project_name}.",
                "task_type": TaskType.DOCUMENTATION,
                "required_capabilities": ["documentation"],
                "estimated_effort": 45
            }
        ]
    
    elif request.project_type == "frontend_app":
        return [
            {
                "title": f"Design {request.project_name} UI/UX",
                "description": f"Design the user interface and user experience for {request.project_name}. Requirements: {request.requirements}",
                "task_type": TaskType.ARCHITECTURE,
                "required_capabilities": ["frontend_development", "ui_design"],
                "estimated_effort": 90
            },
            {
                "title": f"Implement {request.project_name} Components",
                "description": f"Implement the frontend components and functionality for {request.project_name}. Technology: {request.technology_stack}",
                "task_type": TaskType.CODE_GENERATION,
                "required_capabilities": ["frontend_development", "code_generation"],
                "estimated_effort": 150
            },
            {
                "title": f"Add Frontend Tests for {request.project_name}",
                "description": f"Create unit and integration tests for the frontend components of {request.project_name}.",
                "task_type": TaskType.TESTING,
                "required_capabilities": ["testing", "frontend_development"],
                "estimated_effort": 75
            }
        ]
    
    else:
        # Generic project template
        return [
            {
                "title": f"Plan {request.project_name} Implementation",
                "description": f"Create implementation plan for {request.project_name}. Requirements: {request.requirements}",
                "task_type": TaskType.PLANNING,
                "required_capabilities": ["planning", "architecture"],
                "estimated_effort": 30
            },
            {
                "title": f"Implement {request.project_name}",
                "description": f"Implement {request.project_name} according to requirements: {request.requirements}",
                "task_type": TaskType.CODE_GENERATION,
                "required_capabilities": ["code_generation"],
                "estimated_effort": 120
            },
            {
                "title": f"Test {request.project_name}",
                "description": f"Create and run tests for {request.project_name}.",
                "task_type": TaskType.TESTING,
                "required_capabilities": ["testing"],
                "estimated_effort": 60
            }
        ]


def _calculate_estimated_completion(tasks: List[Task]) -> Optional[datetime]:
    """Calculate estimated project completion time."""
    total_effort = sum(task.estimated_effort or 30 for task in tasks)  # Default 30 min per task
    
    # Assume 8 hours of work per day
    days_needed = total_effort / (8 * 60)
    
    return datetime.utcnow() + timedelta(days=days_needed)