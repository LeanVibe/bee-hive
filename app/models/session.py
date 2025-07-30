"""
Session model for LeanVibe Agent Hive 2.0

Represents coordinated development sessions where multiple agents
collaborate on projects with shared context and objectives.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum

from sqlalchemy import Column, String, Text, DateTime, JSON, Enum as SQLEnum, Boolean
from sqlalchemy.sql import func

from ..core.database import Base
from ..core.database_types import DatabaseAgnosticUUID, UUIDArray, StringArray


class SessionStatus(Enum):
    """Session lifecycle status."""
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"


class SessionType(Enum):
    """Types of development sessions."""
    FEATURE_DEVELOPMENT = "feature_development"
    BUG_FIXING = "bug_fixing"
    ARCHITECTURE_REVIEW = "architecture_review"
    CODE_REVIEW = "code_review"
    DEPLOYMENT = "deployment"
    RESEARCH = "research"
    PLANNING = "planning"
    OPTIMIZATION = "optimization"


class Session(Base):
    """
    Represents a coordinated development session in the multi-agent system.
    
    Sessions orchestrate multiple agents working together on a common
    objective, maintaining shared context and coordination state.
    """
    
    __tablename__ = "sessions"
    
    # Primary identification
    id = Column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    
    # Session classification
    session_type = Column(SQLEnum(SessionType), nullable=False, index=True)
    status = Column(SQLEnum(SessionStatus), nullable=False, default=SessionStatus.INACTIVE, index=True)
    
    # Agent coordination
    participant_agents = Column(UUIDArray(), nullable=True, default=list)
    lead_agent_id = Column(DatabaseAgnosticUUID(), nullable=True)
    
    # Session state and context
    state = Column(JSON, nullable=True, default=dict)
    shared_context = Column(JSON, nullable=True, default=dict)
    objectives = Column(StringArray(), nullable=True, default=list)
    
    # tmux integration
    tmux_session_id = Column(String(255), nullable=True, unique=True)
    
    # Configuration and settings
    config = Column(JSON, nullable=True, default=dict)
    auto_consolidate = Column(Boolean, nullable=False, default=True)
    max_duration_hours = Column(String, nullable=True, default="24")
    
    # Progress tracking
    total_tasks = Column(String, nullable=True, default="0")
    completed_tasks = Column(String, nullable=True, default="0")
    failed_tasks = Column(String, nullable=True, default="0")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    paused_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    last_activity = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self) -> str:
        return f"<Session(id={self.id}, name='{self.name}', type='{self.session_type}', status='{self.status}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "session_type": self.session_type.value,
            "status": self.status.value,
            "participant_agents": [str(agent_id) for agent_id in (self.participant_agents or [])],
            "lead_agent_id": str(self.lead_agent_id) if self.lead_agent_id else None,
            "state": self.state,
            "shared_context": self.shared_context,
            "objectives": self.objectives,
            "tmux_session_id": self.tmux_session_id,
            "config": self.config,
            "auto_consolidate": self.auto_consolidate,
            "max_duration_hours": int(self.max_duration_hours or 24),
            "total_tasks": int(self.total_tasks or 0),
            "completed_tasks": int(self.completed_tasks or 0),
            "failed_tasks": int(self.failed_tasks or 0),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "paused_at": self.paused_at.isoformat() if self.paused_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None
        }
    
    def start_session(self, lead_agent_id: Optional[uuid.UUID] = None) -> None:
        """Start the development session."""
        self.status = SessionStatus.ACTIVE
        self.started_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        
        if lead_agent_id:
            self.lead_agent_id = lead_agent_id
    
    def pause_session(self) -> None:
        """Pause the development session."""
        self.status = SessionStatus.PAUSED
        self.paused_at = datetime.utcnow()
    
    def resume_session(self) -> None:
        """Resume a paused session."""
        if self.status == SessionStatus.PAUSED:
            self.status = SessionStatus.ACTIVE
            self.paused_at = None
            self.last_activity = datetime.utcnow()
    
    def complete_session(self, success: bool = True) -> None:
        """Complete the development session."""
        self.status = SessionStatus.COMPLETED if success else SessionStatus.FAILED
        self.completed_at = datetime.utcnow()
    
    def add_participant(self, agent_id: uuid.UUID) -> None:
        """Add an agent to the session."""
        if self.participant_agents is None:
            self.participant_agents = []
        
        if agent_id not in self.participant_agents:
            self.participant_agents.append(agent_id)
            self.last_activity = datetime.utcnow()
    
    def remove_participant(self, agent_id: uuid.UUID) -> None:
        """Remove an agent from the session."""
        if self.participant_agents and agent_id in self.participant_agents:
            self.participant_agents.remove(agent_id)
            self.last_activity = datetime.utcnow()
    
    def update_shared_context(self, key: str, value: Any) -> None:
        """Update shared context information."""
        if self.shared_context is None:
            self.shared_context = {}
        
        self.shared_context[key] = value
        self.last_activity = datetime.utcnow()
    
    def get_shared_context(self, key: str, default: Any = None) -> Any:
        """Get shared context information."""
        if not self.shared_context:
            return default
        
        return self.shared_context.get(key, default)
    
    def add_objective(self, objective: str) -> None:
        """Add a session objective."""
        if self.objectives is None:
            self.objectives = []
        
        if objective not in self.objectives:
            self.objectives.append(objective)
    
    def remove_objective(self, objective: str) -> None:
        """Remove a session objective."""
        if self.objectives and objective in self.objectives:
            self.objectives.remove(objective)
    
    def update_task_counts(self, total: int, completed: int, failed: int) -> None:
        """Update task completion statistics."""
        self.total_tasks = str(total)
        self.completed_tasks = str(completed) 
        self.failed_tasks = str(failed)
        self.last_activity = datetime.utcnow()
    
    def get_progress_percentage(self) -> float:
        """Calculate session progress percentage."""
        total = int(self.total_tasks or 0)
        if total == 0:
            return 0.0
        
        completed = int(self.completed_tasks or 0)
        return (completed / total) * 100.0
    
    def get_success_rate(self) -> float:
        """Calculate task success rate."""
        total_finished = int(self.completed_tasks or 0) + int(self.failed_tasks or 0)
        if total_finished == 0:
            return 100.0
        
        completed = int(self.completed_tasks or 0)
        return (completed / total_finished) * 100.0
    
    def is_active(self) -> bool:
        """Check if session is currently active."""
        return self.status == SessionStatus.ACTIVE
    
    def get_duration_minutes(self) -> Optional[int]:
        """Get session duration in minutes."""
        if not self.started_at:
            return None
        
        end_time = self.completed_at or datetime.utcnow()
        duration = end_time - self.started_at
        return int(duration.total_seconds() / 60)
    
    def should_auto_consolidate(self) -> bool:
        """Check if session should trigger automatic consolidation."""
        if not self.auto_consolidate or not self.started_at:
            return False
        
        max_hours = int(self.max_duration_hours or 24)
        duration = datetime.utcnow() - self.started_at
        return duration.total_seconds() > (max_hours * 3600)
    
    def get_participant_count(self) -> int:
        """Get number of participating agents."""
        return len(self.participant_agents or [])