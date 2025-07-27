"""
Task model for LeanVibe Agent Hive 2.0

Represents development tasks that can be assigned to agents
for coordinated multi-agent development workflows.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from enum import Enum

from sqlalchemy import Column, String, Text, DateTime, JSON, Enum as SQLEnum, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from ..core.database import Base


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    ASSIGNED = "assigned" 
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    MEDIUM = 5
    HIGH = 8
    CRITICAL = 10


class TaskType(Enum):
    """Types of development tasks."""
    FEATURE_DEVELOPMENT = "feature_development"
    BUG_FIX = "bug_fix"
    REFACTORING = "refactoring"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    ARCHITECTURE = "architecture"
    DEPLOYMENT = "deployment"
    CODE_REVIEW = "code_review"
    RESEARCH = "research"
    OPTIMIZATION = "optimization"
    # Additional task types for persona system
    CODE_GENERATION = "code_generation"
    COORDINATION = "coordination"
    PLANNING = "planning"


class Task(Base):
    """
    Represents a development task in the multi-agent system.
    
    Tasks can be assigned to specific agents based on their capabilities
    and can have dependencies on other tasks for complex workflows.
    """
    
    __tablename__ = "tasks"
    
    # Primary identification
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Task classification
    task_type = Column(SQLEnum(TaskType), nullable=True, index=True)
    status = Column(SQLEnum(TaskStatus), nullable=False, default=TaskStatus.PENDING, index=True)
    priority = Column(SQLEnum(TaskPriority), nullable=False, default=TaskPriority.MEDIUM, index=True)
    
    # Assignment and ownership
    assigned_agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"), nullable=True, index=True)
    created_by_agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"), nullable=True)
    
    # Task relationships
    dependencies = Column(ARRAY(UUID(as_uuid=True)), nullable=True, default=list)
    blocking_tasks = Column(ARRAY(UUID(as_uuid=True)), nullable=True, default=list)
    
    # Task context and configuration
    context = Column(JSON, nullable=True, default=dict)
    required_capabilities = Column(ARRAY(String), nullable=True, default=list)
    estimated_effort = Column(Integer, nullable=True)  # in minutes
    actual_effort = Column(Integer, nullable=True)     # in minutes
    
    # Execution results
    result = Column(JSON, nullable=True, default=dict)
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    max_retries = Column(Integer, nullable=False, default=3)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    assigned_at = Column(DateTime(timezone=True), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    due_date = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    assigned_agent = relationship("Agent", foreign_keys=[assigned_agent_id])
    created_by = relationship("Agent", foreign_keys=[created_by_agent_id])
    persona_performance = relationship("PersonaPerformanceModel", back_populates="task")
    
    def __init__(self, **kwargs):
        """Initialize task with proper defaults."""
        # Set default values if not provided
        if 'status' not in kwargs:
            kwargs['status'] = TaskStatus.PENDING
        if 'priority' not in kwargs:
            kwargs['priority'] = TaskPriority.MEDIUM
        if 'retry_count' not in kwargs:
            kwargs['retry_count'] = 0
        if 'max_retries' not in kwargs:
            kwargs['max_retries'] = 3
        if 'dependencies' not in kwargs:
            kwargs['dependencies'] = []
        if 'blocking_tasks' not in kwargs:
            kwargs['blocking_tasks'] = []
        if 'context' not in kwargs:
            kwargs['context'] = {}
        if 'required_capabilities' not in kwargs:
            kwargs['required_capabilities'] = []
        if 'result' not in kwargs:
            kwargs['result'] = {}
        
        super().__init__(**kwargs)
    
    def __repr__(self) -> str:
        return f"<Task(id={self.id}, title='{self.title}', status='{self.status}', priority='{self.priority}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization."""
        return {
            "id": str(self.id),
            "title": self.title,
            "description": self.description,
            "task_type": self.task_type.value if self.task_type else None,
            "status": self.status.value,
            "priority": self.priority.name.lower(),
            "assigned_agent_id": str(self.assigned_agent_id) if self.assigned_agent_id else None,
            "created_by_agent_id": str(self.created_by_agent_id) if self.created_by_agent_id else None,
            "dependencies": [str(dep) for dep in (self.dependencies or [])],
            "blocking_tasks": [str(task) for task in (self.blocking_tasks or [])],
            "context": self.context,
            "required_capabilities": self.required_capabilities,
            "estimated_effort": self.estimated_effort,
            "actual_effort": self.actual_effort,
            "result": self.result,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "assigned_at": self.assigned_at.isoformat() if self.assigned_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "due_date": self.due_date.isoformat() if self.due_date else None
        }
    
    def can_be_started(self, available_tasks: List[str]) -> bool:
        """Check if task can be started based on dependencies."""
        if not self.dependencies:
            return True
        
        # All dependencies must be completed
        return all(str(dep_id) in available_tasks for dep_id in self.dependencies)
    
    def assign_to_agent(self, agent_id: uuid.UUID) -> None:
        """Assign task to a specific agent."""
        self.assigned_agent_id = agent_id
        self.assigned_at = datetime.utcnow()
        self.status = TaskStatus.ASSIGNED
    
    def start_execution(self) -> None:
        """Mark task as started."""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.utcnow()
    
    def complete_successfully(self, result: Dict[str, Any]) -> None:
        """Mark task as completed successfully."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.result = result
        
        # Calculate actual effort if started_at is available
        if self.started_at:
            duration = datetime.utcnow() - self.started_at
            self.actual_effort = int(duration.total_seconds() / 60)
    
    def fail_with_error(self, error_message: str, can_retry: bool = True) -> None:
        """Mark task as failed with error information."""
        self.error_message = error_message
        self.retry_count += 1
        
        if can_retry and self.retry_count < self.max_retries:
            self.status = TaskStatus.PENDING  # Reset for retry
        else:
            self.status = TaskStatus.FAILED
            self.completed_at = datetime.utcnow()
            
            # Calculate actual effort if started_at is available
            if self.started_at:
                duration = datetime.utcnow() - self.started_at
                self.actual_effort = int(duration.total_seconds() / 60)
    
    def block_with_reason(self, reason: str) -> None:
        """Mark task as blocked."""
        self.status = TaskStatus.BLOCKED
        self.error_message = reason
    
    def calculate_urgency_score(self) -> float:
        """Calculate urgency score based on priority and due date."""
        base_score = self.priority.value / 10.0
        
        if self.due_date:
            days_until_due = (self.due_date - datetime.utcnow()).days
            if days_until_due <= 0:
                return 1.0  # Overdue = maximum urgency
            elif days_until_due <= 1:
                return min(1.0, base_score + 0.3)
            elif days_until_due <= 7:
                return min(1.0, base_score + 0.1)
        
        return base_score
    
    def add_dependency(self, task_id: uuid.UUID) -> None:
        """Add a task dependency."""
        if self.dependencies is None:
            self.dependencies = []
        
        if task_id not in self.dependencies:
            self.dependencies.append(task_id)
    
    def remove_dependency(self, task_id: uuid.UUID) -> None:
        """Remove a task dependency."""
        if self.dependencies and task_id in self.dependencies:
            self.dependencies.remove(task_id)
    
    def add_blocking_task(self, task_id: uuid.UUID) -> None:
        """Add a task that this task blocks."""
        if self.blocking_tasks is None:
            self.blocking_tasks = []
        
        if task_id not in self.blocking_tasks:
            self.blocking_tasks.append(task_id)
    
    def get_estimated_completion_time(self) -> Optional[datetime]:
        """Estimate completion time based on effort and current time."""
        if not self.estimated_effort:
            return None
        
        start_time = self.started_at or datetime.utcnow()
        return start_time + timedelta(minutes=self.estimated_effort)