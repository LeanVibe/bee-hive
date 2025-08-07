"""
Test-only model definitions for LeanVibe Agent Hive 2.0

These models are simplified, SQLite-compatible versions of the production models
specifically designed for testing without PostgreSQL-specific features.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, Optional, List
from sqlalchemy import String, Integer, DateTime, JSON, Boolean, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.core.database_types import DatabaseAgnosticUUID, StringArray
from app.models.agent import AgentStatus, AgentType  # Import enums
from app.models.task import TaskStatus, TaskPriority, TaskType
from app.models.session import SessionStatus, SessionType
from app.models.workflow import WorkflowStatus, WorkflowPriority


class TestAgent(Base):
    """Test-compatible Agent model without PostgreSQL dependencies."""
    
    __tablename__ = "agents"
    
    id: Mapped[uuid.UUID] = mapped_column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    type: Mapped[AgentType] = mapped_column(nullable=False, default=AgentType.CLAUDE)
    role: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[AgentStatus] = mapped_column(nullable=False, default=AgentStatus.inactive)
    
    # JSON fields compatible with SQLite
    capabilities: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    config: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    
    # Optional fields
    system_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class TestTask(Base):
    """Test-compatible Task model without PostgreSQL dependencies."""
    
    __tablename__ = "tasks"
    
    id: Mapped[uuid.UUID] = mapped_column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Enum fields
    task_type: Mapped[TaskType] = mapped_column(nullable=False)
    status: Mapped[TaskStatus] = mapped_column(nullable=False, default=TaskStatus.PENDING)
    priority: Mapped[TaskPriority] = mapped_column(nullable=False, default=TaskPriority.MEDIUM)
    
    # Foreign keys
    assigned_agent_id: Mapped[Optional[uuid.UUID]] = mapped_column(DatabaseAgnosticUUID(), nullable=True)
    
    # JSON fields
    context: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    required_capabilities: Mapped[List[str]] = mapped_column(JSON, nullable=False, default=list)
    
    # Numeric fields
    estimated_effort: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class TestSession(Base):
    """Test-compatible Session model without PostgreSQL dependencies."""
    
    __tablename__ = "sessions"
    
    id: Mapped[uuid.UUID] = mapped_column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Enum fields
    session_type: Mapped[SessionType] = mapped_column(nullable=False)
    status: Mapped[SessionStatus] = mapped_column(nullable=False, default=SessionStatus.INACTIVE)
    
    # Foreign keys and arrays
    lead_agent_id: Mapped[Optional[uuid.UUID]] = mapped_column(DatabaseAgnosticUUID(), nullable=True)
    participant_agents: Mapped[List[str]] = mapped_column(JSON, nullable=False, default=list)  # Store as JSON for SQLite
    
    # JSON fields
    objectives: Mapped[List[str]] = mapped_column(JSON, nullable=False, default=list)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class TestWorkflow(Base):
    """Test-compatible Workflow model without PostgreSQL dependencies."""
    
    __tablename__ = "workflows"
    
    id: Mapped[uuid.UUID] = mapped_column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Enum fields
    status: Mapped[WorkflowStatus] = mapped_column(nullable=False, default=WorkflowStatus.CREATED)
    priority: Mapped[WorkflowPriority] = mapped_column(nullable=False, default=WorkflowPriority.MEDIUM)
    
    # JSON fields
    definition: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    context: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    variables: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    dependencies: Mapped[Dict[str, List[str]]] = mapped_column(JSON, nullable=False, default=dict)
    
    # Task management
    task_ids: Mapped[List[str]] = mapped_column(JSON, nullable=False, default=list)
    total_tasks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    
    # Timing
    estimated_duration: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# Map test models to production imports for seamless compatibility
Agent = TestAgent
Task = TestTask
Session = TestSession
Workflow = TestWorkflow