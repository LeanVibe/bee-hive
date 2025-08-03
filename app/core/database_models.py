"""
Database Models for LeanVibe Agent Hive 2.0 Enterprise Operations

Comprehensive SQLAlchemy models for enterprise pilot management, ROI tracking,
executive engagement, and autonomous development operations.

CRITICAL COMPONENT: Provides data layer for all enterprise functionality.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum as PyEnum

import structlog
from sqlalchemy import (
    Column, String, Integer, DateTime, Boolean, Text, 
    ForeignKey, JSON, Enum, Index, UniqueConstraint
)
from sqlalchemy.sql.sqltypes import Numeric
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

from .database import Base

logger = structlog.get_logger()


# Enumerations
class PilotTier(PyEnum):
    """Enterprise pilot tier classification."""
    FORTUNE_50 = "fortune_50"
    FORTUNE_100 = "fortune_100" 
    FORTUNE_500 = "fortune_500"
    ENTERPRISE = "enterprise"


class PilotStatus(PyEnum):
    """Pilot execution status."""
    QUEUED = "queued"
    ACTIVE = "active"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ON_HOLD = "on_hold"
    CANCELLED = "cancelled"


class EngagementType(PyEnum):
    """Executive engagement types."""
    INITIAL_BRIEFING = "initial_briefing"
    WEEKLY_PROGRESS_REVIEW = "weekly_progress_review"
    MILESTONE_CELEBRATION = "milestone_celebration"
    ROI_PRESENTATION = "roi_presentation"
    CONVERSION_DISCUSSION = "conversion_discussion"


class TaskStatus(PyEnum):
    """Development task status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Core Models (Agent model is in app.models.agent)
class EnterprisePilot(Base):
    """Enterprise pilot program model."""
    __tablename__ = "enterprise_pilots"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pilot_id = Column(String(100), unique=True, nullable=False)
    
    # Company information
    company_name = Column(String(255), nullable=False)
    company_tier = Column(Enum(PilotTier), nullable=False)
    industry = Column(String(100), nullable=False)
    annual_revenue = Column(Numeric(15, 2))
    employee_count = Column(Integer)
    
    # Contact information (stored as JSONB for flexibility)
    primary_contact = Column(JSONB, nullable=False)
    technical_contacts = Column(JSONB, default=[])
    executive_contacts = Column(JSONB, default=[])
    
    # Pilot configuration
    use_cases = Column(JSONB, default=[])
    compliance_requirements = Column(JSONB, default=[])
    integration_requirements = Column(JSONB, default=[])
    success_criteria = Column(JSONB, default={})
    
    # Timeline and status
    pilot_start_date = Column(DateTime(timezone=True))
    pilot_end_date = Column(DateTime(timezone=True))
    pilot_duration_weeks = Column(Integer, default=4)
    current_status = Column(Enum(PilotStatus), default=PilotStatus.QUEUED)
    
    # Success tracking
    success_score = Column(Numeric(5, 2), default=0.0)
    stakeholder_satisfaction = Column(Numeric(5, 2), default=0.0)
    technical_success_rate = Column(Numeric(5, 2), default=0.0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    roi_metrics = relationship("ROIMetrics", back_populates="pilot", cascade="all, delete-orphan")
    executive_engagements = relationship("ExecutiveEngagement", back_populates="pilot", cascade="all, delete-orphan")
    demo_sessions = relationship("DemoSession", back_populates="pilot", cascade="all, delete-orphan")
    development_tasks = relationship("DevelopmentTask", back_populates="pilot", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_pilots_company_tier', 'company_tier'),
        Index('idx_pilots_status', 'current_status'),
        Index('idx_pilots_industry', 'industry'),
    )


class ROIMetrics(Base):
    """ROI tracking and measurement model."""
    __tablename__ = "roi_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pilot_id = Column(UUID(as_uuid=True), ForeignKey('enterprise_pilots.id', ondelete='CASCADE'), nullable=False)
    
    # Baseline metrics
    baseline_velocity = Column(Numeric(10, 2), default=0.0)
    baseline_quality_score = Column(Numeric(5, 2), default=0.0)
    baseline_development_cost = Column(Numeric(15, 2), default=0.0)
    
    # Current performance
    current_velocity = Column(Numeric(10, 2), default=0.0)
    current_quality_score = Column(Numeric(5, 2), default=0.0)
    current_development_cost = Column(Numeric(15, 2), default=0.0)
    
    # Improvement calculations
    velocity_improvement_factor = Column(Numeric(10, 2), default=0.0)
    quality_improvement_percentage = Column(Numeric(5, 2), default=0.0)
    cost_savings_percentage = Column(Numeric(5, 2), default=0.0)
    
    # ROI calculations
    total_time_saved_hours = Column(Numeric(10, 2), default=0.0)
    total_cost_savings = Column(Numeric(15, 2), default=0.0)
    roi_percentage = Column(Numeric(10, 2), default=0.0)
    
    # Measurement metadata
    measurement_date = Column(DateTime(timezone=True), server_default=func.now())
    measurement_type = Column(String(50), default="automated")
    validation_status = Column(String(50), default="pending")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    pilot = relationship("EnterprisePilot", back_populates="roi_metrics")
    
    # Indexes
    __table_args__ = (
        Index('idx_roi_pilot_id', 'pilot_id'),
        Index('idx_roi_measurement_date', 'measurement_date'),
    )


class ExecutiveEngagement(Base):
    """Executive engagement and communication tracking."""
    __tablename__ = "executive_engagements"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    engagement_id = Column(String(100), unique=True, nullable=False)
    pilot_id = Column(UUID(as_uuid=True), ForeignKey('enterprise_pilots.id', ondelete='CASCADE'), nullable=False)
    
    # Executive information
    executive_name = Column(String(255), nullable=False)
    executive_title = Column(String(255), nullable=False)
    executive_email = Column(String(255), nullable=False)
    executive_role = Column(String(100), nullable=False)
    
    # Engagement details
    engagement_type = Column(Enum(EngagementType), nullable=False)
    engagement_status = Column(String(50), default="scheduled")
    scheduled_time = Column(DateTime(timezone=True))
    actual_time = Column(DateTime(timezone=True))
    duration_minutes = Column(Integer, default=30)
    
    # Content and outcomes
    agenda_items = Column(JSONB, default=[])
    presentation_materials = Column(JSONB, default=[])
    key_metrics_presented = Column(JSONB, default={})
    engagement_outcomes = Column(JSONB, default=[])
    action_items = Column(JSONB, default=[])
    
    # Success metrics
    satisfaction_score = Column(Numeric(3, 1), default=0.0)
    interest_level = Column(String(20), default="medium")
    conversion_progress = Column(String(50), default="discovery")
    follow_up_required = Column(Boolean, default=False)
    next_engagement_date = Column(DateTime(timezone=True))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    pilot = relationship("EnterprisePilot", back_populates="executive_engagements")
    
    # Indexes
    __table_args__ = (
        Index('idx_engagements_pilot_id', 'pilot_id'),
        Index('idx_engagements_status', 'engagement_status'),
        Index('idx_engagements_scheduled', 'scheduled_time'),
    )


class DemoSession(Base):
    """Demo session tracking and outcomes."""
    __tablename__ = "demo_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(100), unique=True, nullable=False)
    pilot_id = Column(UUID(as_uuid=True), ForeignKey('enterprise_pilots.id', ondelete='CASCADE'), nullable=False)
    
    # Demo configuration
    demo_type = Column(String(100), nullable=False)
    demo_scenario = Column(String(100), nullable=False)
    demo_environment_url = Column(String(500))
    
    # Scheduling
    scheduled_time = Column(DateTime(timezone=True), nullable=False)
    actual_start_time = Column(DateTime(timezone=True))
    actual_end_time = Column(DateTime(timezone=True))
    duration_minutes = Column(Integer, default=30)
    
    # Attendees and execution
    attendees = Column(JSONB, default=[])
    presenter = Column(String(255))
    demo_script = Column(JSONB, default={})
    execution_status = Column(String(50), default="scheduled")
    success_rate = Column(Numeric(5, 2), default=0.0)
    technical_issues = Column(JSONB, default=[])
    
    # Outcomes
    attendee_feedback = Column(JSONB, default={})
    demo_objectives_met = Column(JSONB, default=[])
    follow_up_actions = Column(JSONB, default=[])
    
    # Recording and materials
    recording_url = Column(String(500))
    materials_shared = Column(JSONB, default=[])
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    pilot = relationship("EnterprisePilot", back_populates="demo_sessions")
    
    # Indexes
    __table_args__ = (
        Index('idx_demos_pilot_id', 'pilot_id'),
        Index('idx_demos_scheduled', 'scheduled_time'),
        Index('idx_demos_status', 'execution_status'),
    )


class DevelopmentTask(Base):
    """Development task tracking and performance measurement."""
    __tablename__ = "development_tasks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id = Column(String(100), unique=True, nullable=False)
    pilot_id = Column(UUID(as_uuid=True), ForeignKey('enterprise_pilots.id', ondelete='CASCADE'), nullable=False)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agents.id', ondelete='SET NULL'))
    
    # Task details
    task_type = Column(String(100), nullable=False)
    task_description = Column(Text, nullable=False)
    task_complexity = Column(String(50), default="medium")
    task_priority = Column(String(20), default="medium")
    
    # Requirements and context
    requirements = Column(JSONB, default={})
    context_data = Column(JSONB, default={})
    technical_specifications = Column(JSONB, default={})
    
    # Execution tracking
    status = Column(Enum(TaskStatus), default=TaskStatus.PENDING)
    assigned_at = Column(DateTime(timezone=True))
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    
    # Results and quality
    output_artifacts = Column(JSONB, default=[])
    quality_score = Column(Numeric(5, 2), default=0.0)
    test_coverage = Column(Numeric(5, 2), default=0.0)
    performance_metrics = Column(JSONB, default={})
    
    # Time tracking
    estimated_hours = Column(Numeric(6, 2), default=0.0)
    actual_hours = Column(Numeric(6, 2), default=0.0)
    velocity_improvement_factor = Column(Numeric(10, 2), default=0.0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    pilot = relationship("EnterprisePilot", back_populates="development_tasks")
    # agent relationship removed - Agent model is in app.models.agent
    
    # Indexes
    __table_args__ = (
        Index('idx_tasks_pilot_id', 'pilot_id'),
        Index('idx_tasks_agent_id', 'agent_id'),
        Index('idx_tasks_status', 'status'),
        Index('idx_tasks_type', 'task_type'),
    )


class ContextMemory(Base):
    """Context and memory management with vector embeddings."""
    __tablename__ = "context_memories"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    memory_id = Column(String(100), unique=True, nullable=False)
    pilot_id = Column(UUID(as_uuid=True), ForeignKey('enterprise_pilots.id', ondelete='CASCADE'), nullable=False)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agents.id', ondelete='CASCADE'), nullable=False)
    
    # Memory content
    memory_type = Column(String(100), nullable=False)
    content_summary = Column(Text, nullable=False)
    full_content = Column(JSONB, nullable=False)
    
    # Vector embeddings for semantic search (pgvector)
    embedding = Column(Vector(1536))  # OpenAI embedding dimensions
    
    # Metadata
    importance_score = Column(Numeric(3, 2), default=0.5)
    access_frequency = Column(Integer, default=0)
    last_accessed = Column(DateTime(timezone=True), server_default=func.now())
    
    # Context relationships
    related_memories = Column(ARRAY(UUID(as_uuid=True)), default=[])
    parent_session_id = Column(String(100))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    pilot = relationship("EnterprisePilot")
    # agent relationship removed - Agent model is in app.models.agent
    
    # Indexes
    __table_args__ = (
        Index('idx_memories_pilot_agent', 'pilot_id', 'agent_id'),
        Index('idx_memories_type', 'memory_type'),
        Index('idx_memories_importance', 'importance_score'),
        # Vector similarity search index will be created via migration
    )


class SystemEvent(Base):
    """System events and monitoring."""
    __tablename__ = "system_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_id = Column(String(100), unique=True, nullable=False)
    
    # Event classification
    event_type = Column(String(100), nullable=False)
    event_category = Column(String(50), nullable=False)
    severity_level = Column(String(20), default="info")
    
    # Event details
    event_title = Column(String(255), nullable=False)
    event_description = Column(Text)
    event_data = Column(JSONB, default={})
    
    # Context
    pilot_id = Column(String(100))  # Flexible reference
    agent_id = Column(String(100))  # Flexible reference
    user_id = Column(String(100))
    
    # Metadata
    source_system = Column(String(100), nullable=False)
    correlation_id = Column(String(100))
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_events_type', 'event_type'),
        Index('idx_events_category', 'event_category'),
        Index('idx_events_severity', 'severity_level'),
        Index('idx_events_pilot_id', 'pilot_id'),
        Index('idx_events_created', 'created_at'),
    )


class SystemConfiguration(Base):
    """System configuration and settings."""
    __tablename__ = "system_configuration"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    config_key = Column(String(255), unique=True, nullable=False)
    config_value = Column(JSONB, nullable=False)
    config_type = Column(String(50), default="system")
    description = Column(Text)
    is_encrypted = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_config_key', 'config_key'),
        Index('idx_config_type', 'config_type'),
    )


# Database utility functions
class DatabaseOperations:
    """High-level database operations for enterprise functionality."""
    
    @staticmethod
    async def create_enterprise_pilot(session, pilot_data: Dict[str, Any]) -> EnterprisePilot:
        """Create new enterprise pilot with validation."""
        
        pilot = EnterprisePilot(
            pilot_id=str(uuid.uuid4()),
            **pilot_data
        )
        
        session.add(pilot)
        await session.flush()  # Get the ID without committing
        
        logger.info("Enterprise pilot created", 
                   pilot_id=pilot.pilot_id, 
                   company=pilot.company_name)
        
        return pilot
    
    @staticmethod
    async def record_roi_metrics(session, pilot_id: str, metrics: Dict[str, Any]) -> ROIMetrics:
        """Record ROI metrics for pilot."""
        
        # Find pilot
        pilot = await session.query(EnterprisePilot).filter(
            EnterprisePilot.pilot_id == pilot_id
        ).first()
        
        if not pilot:
            raise ValueError(f"Pilot not found: {pilot_id}")
        
        roi_metrics = ROIMetrics(
            pilot_id=pilot.id,
            **metrics
        )
        
        session.add(roi_metrics)
        await session.flush()
        
        return roi_metrics
    
    @staticmethod
    async def create_executive_engagement(session, engagement_data: Dict[str, Any]) -> ExecutiveEngagement:
        """Create executive engagement record."""
        
        engagement = ExecutiveEngagement(
            engagement_id=str(uuid.uuid4()),
            **engagement_data
        )
        
        session.add(engagement)
        await session.flush()
        
        return engagement


# Export all models for easy import
__all__ = [
    'EnterprisePilot', 'ROIMetrics', 'ExecutiveEngagement',
    'DemoSession', 'DevelopmentTask', 'ContextMemory', 'SystemEvent',
    'SystemConfiguration', 'DatabaseOperations', 'PilotTier', 'PilotStatus',
    'EngagementType', 'TaskStatus'
]