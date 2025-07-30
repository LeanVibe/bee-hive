"""
Workflow model for LeanVibe Agent Hive 2.0

Represents multi-agent workflows with task dependencies and 
coordination for complex development processes.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum

from sqlalchemy import Column, String, Text, DateTime, JSON, Enum as SQLEnum, Integer
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from ..core.database import Base
from ..core.database_types import DatabaseAgnosticUUID, UUIDArray


class WorkflowStatus(Enum):
    """Workflow execution status."""
    CREATED = "created"
    PLANNING = "planning"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowPriority(Enum):
    """Workflow priority levels."""
    LOW = 1
    MEDIUM = 5
    HIGH = 8
    CRITICAL = 10


class Workflow(Base):
    """
    Represents a multi-agent workflow in the development system.
    
    Workflows coordinate multiple tasks across different agents
    with dependency management and execution sequencing.
    """
    
    __tablename__ = "workflows"
    
    # Primary identification
    id = Column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Workflow classification
    status = Column(SQLEnum(WorkflowStatus), nullable=False, default=WorkflowStatus.CREATED, index=True)
    priority = Column(SQLEnum(WorkflowPriority), nullable=False, default=WorkflowPriority.MEDIUM, index=True)
    
    # Workflow definition and execution
    definition = Column(JSON, nullable=False, default=dict)
    task_ids = Column(UUIDArray(), nullable=True, default=list)
    dependencies = Column(JSON, nullable=True, default=dict)  # task_id -> [dependency_task_ids]
    
    # Execution context
    context = Column(JSON, nullable=True, default=dict)
    variables = Column(JSON, nullable=True, default=dict)
    
    # Progress tracking
    total_tasks = Column(Integer, nullable=False, default=0)
    completed_tasks = Column(Integer, nullable=False, default=0)
    failed_tasks = Column(Integer, nullable=False, default=0)
    
    # Execution results
    result = Column(JSON, nullable=True, default=dict)
    error_message = Column(Text, nullable=True)
    
    # Timing and scheduling
    estimated_duration = Column(Integer, nullable=True)  # in minutes
    actual_duration = Column(Integer, nullable=True)     # in minutes
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    due_date = Column(DateTime(timezone=True), nullable=True)
    
    def __init__(self, **kwargs):
        """Initialize workflow with proper defaults."""
        # Set default values if not provided
        if 'status' not in kwargs:
            kwargs['status'] = WorkflowStatus.CREATED
        if 'priority' not in kwargs:
            kwargs['priority'] = WorkflowPriority.MEDIUM
        if 'total_tasks' not in kwargs:
            kwargs['total_tasks'] = 0
        if 'completed_tasks' not in kwargs:
            kwargs['completed_tasks'] = 0
        if 'failed_tasks' not in kwargs:
            kwargs['failed_tasks'] = 0
        if 'definition' not in kwargs:
            kwargs['definition'] = {}
        if 'context' not in kwargs:
            kwargs['context'] = {}
        if 'variables' not in kwargs:
            kwargs['variables'] = {}
        if 'result' not in kwargs:
            kwargs['result'] = {}
        if 'dependencies' not in kwargs:
            kwargs['dependencies'] = {}
        if 'task_ids' not in kwargs:
            kwargs['task_ids'] = []
        
        super().__init__(**kwargs)
    
    def __repr__(self) -> str:
        return f"<Workflow(id={self.id}, name='{self.name}', status='{self.status}', priority='{self.priority}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow to dictionary for serialization."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.name.lower(),
            "definition": self.definition,
            "task_ids": [str(task_id) for task_id in (self.task_ids or [])],
            "dependencies": self.dependencies,
            "context": self.context,
            "variables": self.variables,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "result": self.result,
            "error_message": self.error_message,
            "estimated_duration": self.estimated_duration,
            "actual_duration": self.actual_duration,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "due_date": self.due_date.isoformat() if self.due_date else None
        }
    
    def start_execution(self) -> None:
        """Mark workflow as started."""
        self.status = WorkflowStatus.RUNNING
        self.started_at = datetime.utcnow()
    
    def complete_successfully(self, result: Dict[str, Any]) -> None:
        """Mark workflow as completed successfully."""
        self.status = WorkflowStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.result = result
        
        # Calculate actual duration if started_at is available
        if self.started_at:
            duration = datetime.utcnow() - self.started_at
            self.actual_duration = int(duration.total_seconds() / 60)
    
    def fail_with_error(self, error_message: str) -> None:
        """Mark workflow as failed with error information."""
        self.status = WorkflowStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
        
        # Calculate actual duration if started_at is available
        if self.started_at:
            duration = datetime.utcnow() - self.started_at
            self.actual_duration = int(duration.total_seconds() / 60)
    
    def pause_execution(self) -> None:
        """Pause workflow execution."""
        if self.status == WorkflowStatus.RUNNING:
            self.status = WorkflowStatus.PAUSED
    
    def resume_execution(self) -> None:
        """Resume paused workflow execution."""
        if self.status == WorkflowStatus.PAUSED:
            self.status = WorkflowStatus.RUNNING
    
    def cancel_execution(self, reason: str = None) -> None:
        """Cancel workflow execution."""
        self.status = WorkflowStatus.CANCELLED
        self.completed_at = datetime.utcnow()
        if reason:
            self.error_message = f"Cancelled: {reason}"
        
        # Calculate actual duration if started_at is available
        if self.started_at:
            duration = datetime.utcnow() - self.started_at
            self.actual_duration = int(duration.total_seconds() / 60)
    
    def update_progress(self, completed_tasks: int, failed_tasks: int) -> None:
        """Update workflow progress based on task completion."""
        self.completed_tasks = completed_tasks
        self.failed_tasks = failed_tasks
        
        # Auto-complete if all tasks are done
        if self.total_tasks > 0 and (completed_tasks + failed_tasks) >= self.total_tasks:
            if failed_tasks == 0:
                self.complete_successfully({"all_tasks_completed": True})
            else:
                self.fail_with_error(f"Workflow failed with {failed_tasks} failed tasks")
    
    def get_completion_percentage(self) -> float:
        """Get workflow completion percentage."""
        if self.total_tasks == 0:
            return 0.0
        
        return (self.completed_tasks / self.total_tasks) * 100.0
    
    def add_task(self, task_id: uuid.UUID, dependencies: List[uuid.UUID] = None) -> None:
        """Add a task to the workflow."""
        if self.task_ids is None:
            self.task_ids = []
        
        if task_id not in self.task_ids:
            self.task_ids.append(task_id)
            self.total_tasks = len(self.task_ids)
            
            # Add dependencies
            if dependencies:
                if self.dependencies is None:
                    self.dependencies = {}
                self.dependencies[str(task_id)] = [str(dep) for dep in dependencies]
    
    def remove_task(self, task_id: uuid.UUID) -> None:
        """Remove a task from the workflow."""
        if self.task_ids and task_id in self.task_ids:
            self.task_ids.remove(task_id)
            self.total_tasks = len(self.task_ids)
            
            # Remove from dependencies
            if self.dependencies:
                self.dependencies.pop(str(task_id), None)
                # Remove as dependency from other tasks
                for task, deps in self.dependencies.items():
                    if str(task_id) in deps:
                        deps.remove(str(task_id))
    
    def get_ready_tasks(self, completed_task_ids: List[str]) -> List[str]:
        """Get list of task IDs that are ready to execute based on dependencies."""
        if not self.task_ids or not self.dependencies:
            return [str(task_id) for task_id in (self.task_ids or [])]
        
        ready_tasks = []
        completed_set = set(completed_task_ids)
        
        for task_id in self.task_ids:
            task_id_str = str(task_id)
            
            # Skip if already completed
            if task_id_str in completed_set:
                continue
            
            # Check if all dependencies are completed
            task_deps = self.dependencies.get(task_id_str, [])
            if all(dep in completed_set for dep in task_deps):
                ready_tasks.append(task_id_str)
        
        return ready_tasks
    
    def validate_dependencies(self) -> List[str]:
        """Validate workflow dependencies for circular references."""
        errors = []
        
        if not self.dependencies:
            return errors
        
        # Check for circular dependencies using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(task_id: str) -> bool:
            if task_id in rec_stack:
                return True
            if task_id in visited:
                return False
            
            visited.add(task_id)
            rec_stack.add(task_id)
            
            for dep in self.dependencies.get(task_id, []):
                if has_cycle(dep):
                    return True
            
            rec_stack.remove(task_id)
            return False
        
        for task_id in self.dependencies.keys():
            if task_id not in visited:
                if has_cycle(task_id):
                    errors.append(f"Circular dependency detected involving task {task_id}")
        
        return errors
    
    def estimate_completion_time(self) -> Optional[datetime]:
        """Estimate workflow completion time based on current progress."""
        if not self.started_at or self.total_tasks == 0:
            return None
        
        elapsed_time = datetime.utcnow() - self.started_at
        progress_ratio = self.completed_tasks / self.total_tasks
        
        if progress_ratio > 0:
            estimated_total_time = elapsed_time / progress_ratio
            return self.started_at + estimated_total_time
        
        return None