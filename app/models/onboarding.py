"""
Onboarding Database Models

Database models for Epic 6 interactive onboarding system.
Tracks user onboarding sessions, step completions, events, and analytics
for comprehensive onboarding experience optimization.

Epic 6: Advanced User Experience & Adoption
"""

from datetime import datetime
from typing import Dict, Any, Optional
from uuid import UUID, uuid4

from sqlalchemy import Column, String, DateTime, Integer, Boolean, Text, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import relationship
from ..core.database import Base

class OnboardingSession(Base):
    """
    Main onboarding session tracking.
    
    Tracks complete user onboarding journey from start to completion or abandonment.
    """
    __tablename__ = "onboarding_sessions"

    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    
    # Session timing
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime, nullable=True, index=True)
    skipped_at = Column(DateTime, nullable=True)
    total_time = Column(Integer, nullable=True)  # Total time in milliseconds
    
    # Progress tracking
    current_step = Column(Integer, default=1, nullable=False)  # 1-5
    progress = Column(JSON, default=dict)  # Flexible progress data
    
    # Session metadata
    user_agent = Column(Text, nullable=True)
    referrer = Column(Text, nullable=True)
    source = Column(String(50), default="web")  # web, mobile, api
    
    # Completion/abandonment details
    skip_reason = Column(String(255), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="onboarding_sessions")
    steps = relationship("OnboardingStep", back_populates="session", cascade="all, delete-orphan")
    events = relationship("OnboardingEvent", back_populates="session", cascade="all, delete-orphan")
    metrics = relationship("OnboardingMetric", back_populates="session", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<OnboardingSession(id={self.id}, user_id={self.user_id}, step={self.current_step})>"

    @property
    def is_completed(self) -> bool:
        """Check if onboarding session is completed."""
        return self.completed_at is not None

    @property
    def is_active(self) -> bool:
        """Check if onboarding session is currently active."""
        return self.completed_at is None and self.skipped_at is None

    @property
    def completion_rate(self) -> float:
        """Calculate completion rate based on completed steps."""
        if not self.progress:
            return 0.0
        completed_steps = len(self.progress.get("steps_completed", []))
        return (completed_steps / 5) * 100

    @property
    def time_spent(self) -> Optional[int]:
        """Calculate time spent in session (in milliseconds)."""
        if self.completed_at:
            return self.total_time
        elif self.skipped_at:
            return int((self.skipped_at - self.started_at).total_seconds() * 1000)
        else:
            return int((datetime.utcnow() - self.started_at).total_seconds() * 1000)

class OnboardingStep(Base):
    """
    Individual step completion tracking.
    
    Records each step completion with timing and contextual data.
    """
    __tablename__ = "onboarding_steps"

    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey("onboarding_sessions.id"), nullable=False, index=True)
    
    # Step details
    step_number = Column(Integer, nullable=False)  # 1-5
    step_name = Column(String(100), nullable=True)  # welcome, agent_creation, etc.
    
    # Completion tracking
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    time_spent = Column(Integer, nullable=True)  # Time spent on this step (ms)
    
    # Step-specific data
    step_data = Column(JSON, default=dict)  # Step-specific information
    
    # User actions/interactions on this step
    interactions = Column(JSON, default=dict)  # UI interactions, clicks, etc.
    
    # Relationships
    session = relationship("OnboardingSession", back_populates="steps")

    def __repr__(self):
        return f"<OnboardingStep(id={self.id}, session_id={self.session_id}, step={self.step_number})>"

    @property
    def step_name_display(self) -> str:
        """Get display name for step."""
        step_names = {
            1: "Welcome",
            2: "Agent Creation", 
            3: "Dashboard Tour",
            4: "First Task",
            5: "Completion"
        }
        return step_names.get(self.step_number, f"Step {self.step_number}")

class OnboardingEvent(Base):
    """
    Granular event tracking during onboarding.
    
    Captures specific user interactions, errors, and behavioral events.
    """
    __tablename__ = "onboarding_events"

    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey("onboarding_sessions.id"), nullable=False, index=True)
    
    # Event details
    event_name = Column(String(100), nullable=False, index=True)
    event_category = Column(String(50), nullable=True)  # interaction, error, navigation, etc.
    
    # Event data
    event_data = Column(JSON, default=dict)
    
    # Timing
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Context
    step_number = Column(Integer, nullable=True)  # Which step this event occurred on
    element_id = Column(String(255), nullable=True)  # UI element interacted with
    
    # Relationships
    session = relationship("OnboardingSession", back_populates="events")

    def __repr__(self):
        return f"<OnboardingEvent(id={self.id}, event_name={self.event_name}, session_id={self.session_id})>"

class OnboardingMetric(Base):
    """
    Aggregated metrics and analytics for onboarding optimization.
    
    Stores calculated metrics, A/B test data, and performance indicators.
    """
    __tablename__ = "onboarding_metrics"

    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey("onboarding_sessions.id"), nullable=False, index=True)
    
    # Metric details
    metric_type = Column(String(100), nullable=False, index=True)  # completion, drop_off, engagement, etc.
    metric_name = Column(String(255), nullable=True)
    
    # Metric values
    metric_value = Column(JSON, nullable=True)  # Flexible metric storage
    metric_data = Column(JSON, default=dict)    # Additional metric context
    
    # Categorization
    category = Column(String(50), nullable=True)      # performance, behavior, conversion
    subcategory = Column(String(100), nullable=True)  # More specific categorization
    
    # Timing
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    period_start = Column(DateTime, nullable=True)    # For time-range metrics
    period_end = Column(DateTime, nullable=True)
    
    # A/B Testing
    experiment_id = Column(String(100), nullable=True)  # A/B test identifier
    variant = Column(String(50), nullable=True)         # A/B test variant
    
    # Relationships
    session = relationship("OnboardingSession", back_populates="metrics")

    def __repr__(self):
        return f"<OnboardingMetric(id={self.id}, type={self.metric_type}, session_id={self.session_id})>"

class OnboardingTemplate(Base):
    """
    Onboarding flow templates for different user segments.
    
    Allows for customized onboarding experiences based on user type, goals, etc.
    """
    __tablename__ = "onboarding_templates"

    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Template details
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    version = Column(String(20), default="1.0")
    
    # Template configuration
    template_config = Column(JSON, nullable=False)  # Step configuration, content, etc.
    
    # Targeting
    target_segments = Column(JSON, default=list)    # User segments this template targets
    conditions = Column(JSON, default=dict)         # Conditions for template application
    
    # Status
    is_active = Column(Boolean, default=True)
    is_default = Column(Boolean, default=False)
    
    # Performance tracking
    usage_count = Column(Integer, default=0)
    success_rate = Column(Integer, nullable=True)  # Percentage
    
    def __repr__(self):
        return f"<OnboardingTemplate(id={self.id}, name={self.name}, active={self.is_active})>"

class OnboardingFeedback(Base):
    """
    User feedback collected during or after onboarding.
    
    Captures user satisfaction, pain points, and improvement suggestions.
    """
    __tablename__ = "onboarding_feedback"

    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey("onboarding_sessions.id"), nullable=False, index=True)
    
    # Feedback details
    feedback_type = Column(String(50), nullable=False)  # rating, comment, survey, nps
    rating = Column(Integer, nullable=True)             # 1-5 or 1-10 scale
    comment = Column(Text, nullable=True)
    
    # Context
    step_number = Column(Integer, nullable=True)    # Which step feedback was given for
    category = Column(String(100), nullable=True)  # usability, content, technical, etc.
    
    # Feedback data
    feedback_data = Column(JSON, default=dict)     # Structured feedback responses
    
    # Timing
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Follow-up
    requires_follow_up = Column(Boolean, default=False)
    follow_up_notes = Column(Text, nullable=True)
    
    # Relationships
    session = relationship("OnboardingSession")

    def __repr__(self):
        return f"<OnboardingFeedback(id={self.id}, type={self.feedback_type}, rating={self.rating})>"

# Add relationships to User model if needed
# This should be added to the User model in user.py:
# onboarding_sessions = relationship("OnboardingSession", back_populates="user", cascade="all, delete-orphan")