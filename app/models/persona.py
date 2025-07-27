"""
Database models for Agent Persona System.

Defines persistence layer for persona definitions, assignments,
and performance tracking with full SQLAlchemy integration.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from sqlalchemy import Column, String, Text, DateTime, Boolean, Float, Integer, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property

from .base import BaseModel


class PersonaDefinitionModel(BaseModel):
    """Database model for persona definitions."""
    
    __tablename__ = "persona_definitions"
    
    # Primary identification
    id = Column(String(100), primary_key=True, index=True)
    name = Column(String(200), nullable=False, index=True)
    description = Column(Text, nullable=False)
    
    # Persona classification
    persona_type = Column(String(50), nullable=False, index=True)
    adaptation_mode = Column(String(30), nullable=False, default="adaptive")
    
    # Core capabilities and preferences (stored as JSON)
    capabilities = Column(JSONB, nullable=False, default=dict)
    preferred_task_types = Column(JSONB, nullable=False, default=list)
    expertise_domains = Column(JSONB, nullable=False, default=list)
    
    # Behavioral characteristics
    communication_style = Column(JSONB, nullable=False, default=dict)
    decision_making_style = Column(JSONB, nullable=False, default=dict)
    problem_solving_approach = Column(JSONB, nullable=False, default=dict)
    
    # Collaboration preferences
    min_team_size = Column(Integer, nullable=False, default=1)
    max_team_size = Column(Integer, nullable=False, default=8)
    collaboration_patterns = Column(JSONB, nullable=False, default=list)
    mentoring_capability = Column(Boolean, nullable=False, default=False)
    
    # Performance characteristics
    typical_response_time = Column(Float, nullable=False, default=90.0)
    accuracy_vs_speed_preference = Column(Float, nullable=False, default=0.65)
    risk_tolerance = Column(Float, nullable=False, default=0.5)
    
    # Metadata
    version = Column(String(20), nullable=False, default="1.0.0")
    active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    assignments = relationship("PersonaAssignmentModel", back_populates="persona_definition")
    performance_records = relationship("PersonaPerformanceModel", back_populates="persona_definition")
    
    @hybrid_property
    def preferred_team_size_range(self) -> tuple:
        """Get team size range as tuple."""
        return (self.min_team_size, self.max_team_size)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "persona_type": self.persona_type,
            "adaptation_mode": self.adaptation_mode,
            "capabilities": self.capabilities,
            "preferred_task_types": self.preferred_task_types,
            "expertise_domains": self.expertise_domains,
            "communication_style": self.communication_style,
            "decision_making_style": self.decision_making_style,
            "problem_solving_approach": self.problem_solving_approach,
            "preferred_team_size": (self.min_team_size, self.max_team_size),
            "collaboration_patterns": self.collaboration_patterns,
            "mentoring_capability": self.mentoring_capability,
            "typical_response_time": self.typical_response_time,
            "accuracy_vs_speed_preference": self.accuracy_vs_speed_preference,
            "risk_tolerance": self.risk_tolerance,
            "version": self.version,
            "active": self.active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None
        }


class PersonaAssignmentModel(BaseModel):
    """Database model for persona assignments to agents."""
    
    __tablename__ = "persona_assignments"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Assignment details
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"), nullable=False, index=True)
    persona_id = Column(String(100), ForeignKey("persona_definitions.id"), nullable=False, index=True)
    session_id = Column(String(100), nullable=False, index=True)
    
    # Assignment metadata
    assigned_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    assignment_reason = Column(String(100), nullable=False)
    confidence_score = Column(Float, nullable=False)
    
    # Performance tracking
    tasks_completed = Column(Integer, nullable=False, default=0)
    success_rate = Column(Float, nullable=False, default=0.0)
    avg_completion_time = Column(Float, nullable=False, default=0.0)
    
    # Context adaptations
    active_adaptations = Column(JSONB, nullable=False, default=dict)
    
    # Status tracking
    active = Column(Boolean, nullable=False, default=True)
    ended_at = Column(DateTime, nullable=True)
    end_reason = Column(String(100), nullable=True)
    
    # Relationships
    agent = relationship("Agent", back_populates="persona_assignments")
    persona_definition = relationship("PersonaDefinitionModel", back_populates="assignments")
    performance_records = relationship("PersonaPerformanceModel", back_populates="assignment")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "agent_id": str(self.agent_id),
            "persona_id": self.persona_id,
            "session_id": self.session_id,
            "assigned_at": self.assigned_at.isoformat() if self.assigned_at else None,
            "assignment_reason": self.assignment_reason,
            "confidence_score": self.confidence_score,
            "tasks_completed": self.tasks_completed,
            "success_rate": self.success_rate,
            "avg_completion_time": self.avg_completion_time,
            "active_adaptations": self.active_adaptations,
            "active": self.active,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "end_reason": self.end_reason
        }


class PersonaPerformanceModel(BaseModel):
    """Database model for tracking persona performance over time."""
    
    __tablename__ = "persona_performance"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Foreign keys
    persona_id = Column(String(100), ForeignKey("persona_definitions.id"), nullable=False, index=True)
    assignment_id = Column(UUID(as_uuid=True), ForeignKey("persona_assignments.id"), nullable=False, index=True)
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"), nullable=True, index=True)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"), nullable=False, index=True)
    
    # Performance metrics
    task_success = Column(Boolean, nullable=False)
    completion_time = Column(Float, nullable=False)
    complexity_score = Column(Float, nullable=False, default=0.5)
    
    # Capability tracking
    capabilities_used = Column(JSONB, nullable=False, default=list)
    capability_performance = Column(JSONB, nullable=False, default=dict)
    
    # Context information
    task_type = Column(String(50), nullable=True, index=True)
    context_data = Column(JSONB, nullable=False, default=dict)
    adaptations_applied = Column(JSONB, nullable=False, default=dict)
    
    # Quality metrics
    quality_score = Column(Float, nullable=True)  # If available
    efficiency_score = Column(Float, nullable=True)  # If available
    collaboration_score = Column(Float, nullable=True)  # If available
    
    # Timestamps
    recorded_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    task_started_at = Column(DateTime, nullable=True)
    task_completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    persona_definition = relationship("PersonaDefinitionModel", back_populates="performance_records")
    assignment = relationship("PersonaAssignmentModel", back_populates="performance_records")
    task = relationship("Task", back_populates="persona_performance")
    agent = relationship("Agent", back_populates="persona_performance")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "persona_id": self.persona_id,
            "assignment_id": str(self.assignment_id),
            "task_id": str(self.task_id) if self.task_id else None,
            "agent_id": str(self.agent_id),
            "task_success": self.task_success,
            "completion_time": self.completion_time,
            "complexity_score": self.complexity_score,
            "capabilities_used": self.capabilities_used,
            "capability_performance": self.capability_performance,
            "task_type": self.task_type,
            "context_data": self.context_data,
            "adaptations_applied": self.adaptations_applied,
            "quality_score": self.quality_score,
            "efficiency_score": self.efficiency_score,
            "collaboration_score": self.collaboration_score,
            "recorded_at": self.recorded_at.isoformat() if self.recorded_at else None,
            "task_started_at": self.task_started_at.isoformat() if self.task_started_at else None,
            "task_completed_at": self.task_completed_at.isoformat() if self.task_completed_at else None
        }


class PersonaCapabilityHistoryModel(BaseModel):
    """Database model for tracking capability evolution over time."""
    
    __tablename__ = "persona_capability_history"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Foreign keys
    persona_id = Column(String(100), ForeignKey("persona_definitions.id"), nullable=False, index=True)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"), nullable=False, index=True)
    
    # Capability details
    capability_name = Column(String(100), nullable=False, index=True)
    capability_level = Column(String(30), nullable=False)
    proficiency_score = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)
    
    # Usage statistics
    usage_count = Column(Integer, nullable=False, default=0)
    success_rate = Column(Float, nullable=False, default=0.0)
    last_used_at = Column(DateTime, nullable=True)
    
    # Change tracking
    previous_proficiency = Column(Float, nullable=True)
    proficiency_change = Column(Float, nullable=False, default=0.0)
    change_reason = Column(String(100), nullable=True)
    
    # Timestamps
    recorded_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Relationships
    persona_definition = relationship("PersonaDefinitionModel")
    agent = relationship("Agent")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "persona_id": self.persona_id,
            "agent_id": str(self.agent_id),
            "capability_name": self.capability_name,
            "capability_level": self.capability_level,
            "proficiency_score": self.proficiency_score,
            "confidence_score": self.confidence_score,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "previous_proficiency": self.previous_proficiency,
            "proficiency_change": self.proficiency_change,
            "change_reason": self.change_reason,
            "recorded_at": self.recorded_at.isoformat() if self.recorded_at else None
        }


class PersonaAnalyticsModel(BaseModel):
    """Database model for storing aggregated persona analytics."""
    
    __tablename__ = "persona_analytics"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Analytics scope
    persona_id = Column(String(100), ForeignKey("persona_definitions.id"), nullable=True, index=True)
    analytics_type = Column(String(50), nullable=False, index=True)  # daily, weekly, monthly
    time_period_start = Column(DateTime, nullable=False, index=True)
    time_period_end = Column(DateTime, nullable=False, index=True)
    
    # Performance metrics
    total_assignments = Column(Integer, nullable=False, default=0)
    total_tasks = Column(Integer, nullable=False, default=0)
    success_rate = Column(Float, nullable=False, default=0.0)
    avg_completion_time = Column(Float, nullable=False, default=0.0)
    
    # Quality metrics
    avg_quality_score = Column(Float, nullable=True)
    avg_efficiency_score = Column(Float, nullable=True)
    avg_collaboration_score = Column(Float, nullable=True)
    
    # Capability metrics
    capability_improvements = Column(JSONB, nullable=False, default=dict)
    most_used_capabilities = Column(JSONB, nullable=False, default=list)
    capability_success_rates = Column(JSONB, nullable=False, default=dict)
    
    # Assignment patterns
    preferred_task_types = Column(JSONB, nullable=False, default=list)
    avg_team_size = Column(Float, nullable=True)
    collaboration_frequency = Column(Float, nullable=False, default=0.0)
    
    # Trends and insights
    performance_trend = Column(String(20), nullable=True)  # improving, declining, stable
    trend_confidence = Column(Float, nullable=False, default=0.0)
    insights = Column(JSONB, nullable=False, default=list)
    recommendations = Column(JSONB, nullable=False, default=list)
    
    # Metadata
    generated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    data_points = Column(Integer, nullable=False, default=0)
    
    # Relationships
    persona_definition = relationship("PersonaDefinitionModel")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "persona_id": self.persona_id,
            "analytics_type": self.analytics_type,
            "time_period_start": self.time_period_start.isoformat() if self.time_period_start else None,
            "time_period_end": self.time_period_end.isoformat() if self.time_period_end else None,
            "total_assignments": self.total_assignments,
            "total_tasks": self.total_tasks,
            "success_rate": self.success_rate,
            "avg_completion_time": self.avg_completion_time,
            "avg_quality_score": self.avg_quality_score,
            "avg_efficiency_score": self.avg_efficiency_score,
            "avg_collaboration_score": self.avg_collaboration_score,
            "capability_improvements": self.capability_improvements,
            "most_used_capabilities": self.most_used_capabilities,
            "capability_success_rates": self.capability_success_rates,
            "preferred_task_types": self.preferred_task_types,
            "avg_team_size": self.avg_team_size,
            "collaboration_frequency": self.collaboration_frequency,
            "performance_trend": self.performance_trend,
            "trend_confidence": self.trend_confidence,
            "insights": self.insights,
            "recommendations": self.recommendations,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "data_points": self.data_points
        }