"""Pydantic schemas for Task API endpoints."""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, ConfigDict

from ..models.task import TaskStatus, TaskPriority, TaskType


class TaskCreate(BaseModel):
    """Schema for creating new tasks."""
    title: str = Field(..., min_length=1, max_length=255, description="Task title")
    description: Optional[str] = Field(None, description="Task description")
    task_type: Optional[TaskType] = Field(None, description="Type of task")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="Task priority")
    required_capabilities: Optional[List[str]] = Field(default_factory=list, description="Required capabilities")
    estimated_effort: Optional[int] = Field(None, gt=0, description="Estimated effort in minutes")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Task context")


class TaskUpdate(BaseModel):
    """Schema for updating existing tasks."""
    title: Optional[str] = Field(None, min_length=1, max_length=255, description="Task title")
    description: Optional[str] = Field(None, description="Task description")
    status: Optional[TaskStatus] = Field(None, description="Task status")
    priority: Optional[TaskPriority] = Field(None, description="Task priority")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class TaskResponse(BaseModel):
    """Schema for task API responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: uuid.UUID
    title: str
    description: Optional[str]
    task_type: Optional[TaskType]
    status: TaskStatus
    priority: TaskPriority
    assigned_agent_id: Optional[str]
    required_capabilities: Optional[List[str]]
    estimated_effort: Optional[int]
    actual_effort: Optional[int]
    result: Optional[Dict[str, Any]]
    error_message: Optional[str]
    retry_count: int
    created_at: Optional[datetime]
    assigned_at: Optional[datetime]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]


class TaskListResponse(BaseModel):
    """Schema for paginated task list responses."""
    tasks: List[TaskResponse]
    total: int
    offset: int
    limit: int