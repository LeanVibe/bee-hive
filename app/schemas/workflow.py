"""
Pydantic schemas for Workflow API endpoints.
"""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, ConfigDict

from ..models.workflow import WorkflowStatus, WorkflowPriority, WorkflowType


class WorkflowCreate(BaseModel):
    """Schema for creating new workflows."""
    name: str = Field(..., min_length=1, max_length=255, description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    priority: WorkflowPriority = Field(default=WorkflowPriority.MEDIUM, description="Workflow priority")
    workflow_type: WorkflowType = Field(default=WorkflowType.DEVELOPMENT, description="Workflow type")
    definition: Dict[str, Any] = Field(default_factory=dict, description="Workflow definition")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Workflow execution context")
    variables: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Workflow variables")
    estimated_duration: Optional[int] = Field(None, ge=1, description="Estimated duration in minutes")
    due_date: Optional[datetime] = Field(None, description="Workflow due date")


class WorkflowUpdate(BaseModel):
    """Schema for updating existing workflows."""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    priority: Optional[WorkflowPriority] = Field(None, description="Workflow priority")
    workflow_type: Optional[WorkflowType] = Field(None, description="Workflow type")
    definition: Optional[Dict[str, Any]] = Field(None, description="Workflow definition")
    context: Optional[Dict[str, Any]] = Field(None, description="Workflow execution context")
    variables: Optional[Dict[str, Any]] = Field(None, description="Workflow variables")
    status: Optional[WorkflowStatus] = Field(None, description="Workflow status")
    estimated_duration: Optional[int] = Field(None, ge=1, description="Estimated duration in minutes")
    due_date: Optional[datetime] = Field(None, description="Workflow due date")


class WorkflowTaskAssignment(BaseModel):
    """Schema for assigning tasks to workflows."""
    task_id: uuid.UUID = Field(..., description="Task ID to assign")
    dependencies: Optional[List[uuid.UUID]] = Field(default_factory=list, description="Task dependencies")


class WorkflowExecutionRequest(BaseModel):
    """Schema for workflow execution requests."""
    context_override: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Override workflow context")
    variables_override: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Override workflow variables")


class WorkflowResponse(BaseModel):
    """Schema for workflow API responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: uuid.UUID
    name: str
    description: Optional[str]
    status: WorkflowStatus
    priority: WorkflowPriority
    workflow_type: WorkflowType
    definition: Dict[str, Any]
    task_ids: List[uuid.UUID]
    dependencies: Dict[str, List[str]]
    context: Dict[str, Any]
    variables: Dict[str, Any]
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    result: Dict[str, Any]
    error_message: Optional[str]
    estimated_duration: Optional[int]
    actual_duration: Optional[int]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    due_date: Optional[datetime]


class WorkflowListResponse(BaseModel):
    """Schema for paginated workflow list responses."""
    workflows: List[WorkflowResponse]
    total: int
    offset: int
    limit: int


class WorkflowProgressResponse(BaseModel):
    """Schema for workflow progress responses."""
    workflow_id: uuid.UUID
    name: str
    status: WorkflowStatus
    completion_percentage: float
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    pending_tasks: int
    estimated_completion: Optional[datetime]
    current_tasks: List[uuid.UUID]
    ready_tasks: List[uuid.UUID]


class WorkflowValidationResponse(BaseModel):
    """Schema for workflow validation responses."""
    workflow_id: uuid.UUID
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class WorkflowStatsResponse(BaseModel):
    """Schema for workflow statistics."""
    workflow_id: uuid.UUID
    name: str
    total_execution_time: int  # in minutes
    average_task_time: float   # in minutes
    success_rate: float        # percentage
    efficiency_score: float    # 0.0 to 1.0
    bottleneck_tasks: List[uuid.UUID]
    critical_path_duration: int  # in minutes