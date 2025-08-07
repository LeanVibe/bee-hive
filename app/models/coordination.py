"""
Coordination model for LeanVibe Agent Hive 2.0 Enhanced Multi-Agent System

Represents coordination events, collaboration contexts, and agent interactions
in the enhanced multi-agent coordination system for dashboard integration.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

from sqlalchemy import Column, String, Text, DateTime, JSON, Integer, Float, Enum as SQLEnum, ForeignKey
from sqlalchemy.dialects.postgresql import ENUM, UUID, ARRAY
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from ..core.database import Base
from ..core.database_types import DatabaseAgnosticUUID


class CoordinationEventType(Enum):
    """Types of coordination events in the multi-agent system."""
    COLLABORATION_STARTED = "collaboration_started"
    TASK_HANDOFF = "task_handoff" 
    CODE_REVIEW_CYCLE = "code_review_cycle"
    PAIR_PROGRAMMING = "pair_programming"
    KNOWLEDGE_SHARING = "knowledge_sharing"
    DESIGN_REVIEW = "design_review"
    CONFLICT_RESOLUTION = "conflict_resolution"
    TEAM_STANDUP = "team_standup"
    CONTINUOUS_INTEGRATION = "continuous_integration"
    DECISION_MADE = "decision_made"
    PATTERN_COMPLETED = "pattern_completed"


class CoordinationPatternType(Enum):
    """Types of coordination patterns."""
    PAIR_PROGRAMMING = "pair_programming"
    CODE_REVIEW_CYCLE = "code_review_cycle"
    CONTINUOUS_INTEGRATION = "continuous_integration"
    KNOWLEDGE_SHARING = "knowledge_sharing"
    DESIGN_REVIEW = "design_review"
    TASK_HANDOFF = "task_handoff"
    CONFLICT_RESOLUTION = "conflict_resolution"
    TEAM_STANDUP = "team_standup"


class SpecializedAgentRole(Enum):
    """Specialized agent roles for enhanced coordination."""
    ARCHITECT = "architect"
    DEVELOPER = "developer"
    TESTER = "tester"
    REVIEWER = "reviewer"
    DEVOPS = "devops"
    PRODUCT = "product"


class TaskComplexity(Enum):
    """Task complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ENTERPRISE = "enterprise"


class EnhancedCoordinationEvent(Base):
    """
    Represents a coordination event in the multi-agent system.
    
    Tracks all collaboration activities, task handoffs, and inter-agent
    communication for dashboard visibility and business value calculation.
    """
    
    __tablename__ = "enhanced_coordination_events"
    
    # Primary identification
    id = Column(DatabaseAgnosticUUID, primary_key=True, default=uuid.uuid4)
    event_type = Column(SQLEnum(CoordinationEventType), nullable=False, index=True)
    session_id = Column(DatabaseAgnosticUUID, ForeignKey('sessions.id'), nullable=True, index=True)
    
    # Participants and context
    participants = Column(ARRAY(String), nullable=False)  # Agent IDs involved
    primary_agent_id = Column(String, nullable=True, index=True)  # Main agent for the event
    collaboration_id = Column(String, nullable=True, index=True)  # Groups related events
    
    # Event data
    event_data = Column(JSON, nullable=False, default=dict)  # Event-specific data
    collaboration_context = Column(JSON, nullable=True)  # Shared collaboration context
    communication_history = Column(JSON, nullable=True, default=list)  # Communication logs
    decisions_made = Column(JSON, nullable=True, default=list)  # Decisions from this event
    artifacts_created = Column(ARRAY(String), nullable=True)  # Created artifacts/files
    
    # Performance metrics
    duration_ms = Column(Integer, nullable=True)  # Event duration in milliseconds
    success_rate = Column(Float, nullable=True, default=1.0)  # Success rate (0.0-1.0)
    productivity_impact = Column(Float, nullable=True)  # Measured productivity impact
    business_value = Column(Float, nullable=True)  # Calculated business value
    
    # Status and lifecycle
    status = Column(String, nullable=False, default="active")
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    session = relationship("Session")
    
    def __repr__(self):
        return f"<EnhancedCoordinationEvent(id={self.id}, type={self.event_type}, participants={len(self.participants)})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "event_type": self.event_type.value,
            "session_id": str(self.session_id) if self.session_id else None,
            "participants": self.participants,
            "primary_agent_id": self.primary_agent_id,
            "collaboration_id": self.collaboration_id,
            "event_data": self.event_data,
            "collaboration_context": self.collaboration_context,
            "communication_history": self.communication_history,
            "decisions_made": self.decisions_made,
            "artifacts_created": self.artifacts_created,
            "duration_ms": self.duration_ms,
            "success_rate": self.success_rate,
            "productivity_impact": self.productivity_impact,
            "business_value": self.business_value,
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


class CoordinationPattern(Base):
    """
    Defines reusable coordination patterns for multi-agent workflows.
    
    Templates for common collaboration scenarios that can be instantiated
    and tracked for continuous improvement and business value measurement.
    """
    
    __tablename__ = "coordination_patterns"
    
    # Primary identification
    id = Column(DatabaseAgnosticUUID, primary_key=True, default=uuid.uuid4)
    pattern_id = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    
    # Pattern definition
    pattern_type = Column(SQLEnum(CoordinationPatternType), nullable=False, index=True)
    required_roles = Column(ARRAY(String), nullable=False)  # SpecializedAgentRole values
    complexity_level = Column(SQLEnum(TaskComplexity), nullable=False, index=True)
    
    # Pattern configuration
    coordination_steps = Column(JSON, nullable=False, default=list)
    success_metrics = Column(JSON, nullable=False, default=dict)
    estimated_duration_minutes = Column(Integer, nullable=False, default=60)
    
    # Performance tracking
    usage_count = Column(Integer, nullable=False, default=0)
    success_count = Column(Integer, nullable=False, default=0)
    average_duration_ms = Column(Integer, nullable=True)
    average_business_value = Column(Float, nullable=True)
    
    # Lifecycle
    is_active = Column(String, nullable=False, default="active")
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<CoordinationPattern(id={self.pattern_id}, type={self.pattern_type}, usage={self.usage_count})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "pattern_id": self.pattern_id,
            "name": self.name,
            "description": self.description,
            "pattern_type": self.pattern_type.value,
            "required_roles": self.required_roles,
            "complexity_level": self.complexity_level.value,
            "coordination_steps": self.coordination_steps,
            "success_metrics": self.success_metrics,
            "estimated_duration_minutes": self.estimated_duration_minutes,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "average_duration_ms": self.average_duration_ms,
            "average_business_value": self.average_business_value,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate for this pattern."""
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count


class AgentCollaboration(Base):
    """
    Tracks ongoing collaborations between agents.
    
    Represents active collaboration sessions with shared context,
    progress tracking, and business value measurement.
    """
    
    __tablename__ = "agent_collaborations"
    
    # Primary identification
    id = Column(DatabaseAgnosticUUID, primary_key=True, default=uuid.uuid4)
    collaboration_id = Column(String, unique=True, nullable=False, index=True)
    session_id = Column(DatabaseAgnosticUUID, ForeignKey('sessions.id'), nullable=True, index=True)
    pattern_id = Column(String, ForeignKey('coordination_patterns.pattern_id'), nullable=True, index=True)
    
    # Collaboration details
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    participants = Column(ARRAY(String), nullable=False)  # Agent IDs
    primary_agent_id = Column(String, nullable=True)
    
    # Collaboration state
    shared_knowledge = Column(JSON, nullable=False, default=dict)
    communication_history = Column(JSON, nullable=False, default=list)
    decisions_made = Column(JSON, nullable=False, default=list)
    artifacts_created = Column(ARRAY(String), nullable=True)
    success_patterns = Column(JSON, nullable=False, default=list)
    
    # Progress tracking
    status = Column(String, nullable=False, default="active")  # active, completed, failed, paused
    progress_percentage = Column(Float, nullable=False, default=0.0)
    current_step = Column(String, nullable=True)
    
    # Performance metrics
    start_time = Column(DateTime(timezone=True), nullable=False, default=func.now())
    estimated_completion = Column(DateTime(timezone=True), nullable=True)
    actual_completion = Column(DateTime(timezone=True), nullable=True)
    productivity_gain = Column(Float, nullable=True)  # Measured productivity improvement
    business_value_generated = Column(Float, nullable=True)  # Calculated business value
    
    # Metadata
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    
    # Relationships
    session = relationship("Session", back_populates="collaborations")
    pattern = relationship("CoordinationPattern")
    events = relationship("EnhancedCoordinationEvent", 
                         primaryjoin="EnhancedCoordinationEvent.collaboration_id == AgentCollaboration.collaboration_id",
                         foreign_keys="EnhancedCoordinationEvent.collaboration_id")
    
    def __repr__(self):
        return f"<AgentCollaboration(id={self.collaboration_id}, participants={len(self.participants)}, status={self.status})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "collaboration_id": self.collaboration_id,
            "session_id": str(self.session_id) if self.session_id else None,
            "pattern_id": self.pattern_id,
            "name": self.name,
            "description": self.description,
            "participants": self.participants,
            "primary_agent_id": self.primary_agent_id,
            "shared_knowledge": self.shared_knowledge,
            "communication_history": self.communication_history,
            "decisions_made": self.decisions_made,
            "artifacts_created": self.artifacts_created,
            "success_patterns": self.success_patterns,
            "status": self.status,
            "progress_percentage": self.progress_percentage,
            "current_step": self.current_step,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "estimated_completion": self.estimated_completion.isoformat() if self.estimated_completion else None,
            "actual_completion": self.actual_completion.isoformat() if self.actual_completion else None,
            "productivity_gain": self.productivity_gain,
            "business_value_generated": self.business_value_generated,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }