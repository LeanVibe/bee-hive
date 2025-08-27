"""
Business Intelligence Models for LeanVibe Agent Hive 2.0

Database models for comprehensive business analytics and insights:
- Business metrics and KPIs
- User behavior tracking
- Agent performance analytics
- Predictive business modeling data

Epic 5: Business Intelligence & Analytics Engine
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum
from decimal import Decimal

from sqlalchemy import Column, String, Text, DateTime, JSON, Enum as SQLEnum, Integer, Numeric, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from ..core.database import Base
from ..core.database_types import DatabaseAgnosticUUID, UUIDArray, StringArray


class MetricType(Enum):
    """Type of business metric."""
    REVENUE = "revenue"
    USER_ACQUISITION = "user_acquisition"
    SYSTEM_PERFORMANCE = "system_performance"
    AGENT_UTILIZATION = "agent_utilization"
    CUSTOMER_SATISFACTION = "customer_satisfaction"
    CONVERSION = "conversion"
    RETENTION = "retention"
    ENGAGEMENT = "engagement"
    EFFICIENCY = "efficiency"
    GROWTH = "growth"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class BusinessMetric(Base):
    """Core business metrics table for real-time KPI tracking."""
    
    __tablename__ = "business_metrics"
    
    id = Column(DatabaseAgnosticUUID, primary_key=True, default=uuid.uuid4)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_type = Column(SQLEnum(MetricType), nullable=False, index=True)
    metric_value = Column(Numeric(12, 4), nullable=False)
    percentage_value = Column(Numeric(5, 2))  # For percentage-based metrics
    target_value = Column(Numeric(12, 4))  # Target/goal value
    
    # Time-series data
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    period_start = Column(DateTime(timezone=True))
    period_end = Column(DateTime(timezone=True))
    
    # Contextual metadata
    metric_metadata = Column(JSONB)
    tags = Column(StringArray())
    source_system = Column(String(50))  # Which component generated this metric
    
    # Relationships
    agent_id = Column(DatabaseAgnosticUUID, ForeignKey("agents.id"), nullable=True, index=True)
    user_id = Column(DatabaseAgnosticUUID, ForeignKey("users.id"), nullable=True, index=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class UserSession(Base):
    """User session tracking for behavior analytics."""
    
    __tablename__ = "user_sessions"
    
    id = Column(DatabaseAgnosticUUID, primary_key=True, default=uuid.uuid4)
    user_id = Column(DatabaseAgnosticUUID, ForeignKey("users.id"), nullable=True, index=True)
    session_id = Column(String(100), nullable=False, unique=True, index=True)
    
    # Session timing
    session_start = Column(DateTime(timezone=True), nullable=False, index=True)
    session_end = Column(DateTime(timezone=True), index=True)
    duration_seconds = Column(Integer)
    
    # User behavior metrics
    actions_count = Column(Integer, default=0)
    pages_visited = Column(JSONB)
    features_used = Column(StringArray())
    tasks_created = Column(Integer, default=0)
    tasks_completed = Column(Integer, default=0)
    agents_interacted = Column(UUIDArray())
    
    # Session context
    user_agent = Column(Text)
    ip_address = Column(String(45))  # IPv6 compatible
    referrer = Column(Text)
    platform = Column(String(50))  # mobile, desktop, tablet
    device_type = Column(String(50))
    
    # Session quality metrics
    bounce_session = Column(Boolean, default=False)
    conversion_events = Column(JSONB)  # Track conversion actions
    satisfaction_score = Column(Numeric(3, 2))  # 1-5 scale
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class UserJourneyEvent(Base):
    """Individual user journey events for behavior tracking."""
    
    __tablename__ = "user_journey_events"
    
    id = Column(DatabaseAgnosticUUID, primary_key=True, default=uuid.uuid4)
    session_id = Column(DatabaseAgnosticUUID, ForeignKey("user_sessions.id"), nullable=False, index=True)
    user_id = Column(DatabaseAgnosticUUID, ForeignKey("users.id"), nullable=True, index=True)
    
    # Event details
    event_type = Column(String(50), nullable=False, index=True)  # login, create_agent, assign_task, etc.
    event_name = Column(String(100), nullable=False)
    event_category = Column(String(50), index=True)  # onboarding, task_management, agent_interaction
    
    # Event timing
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    sequence_number = Column(Integer, nullable=False)  # Order within session
    
    # Event context
    page_path = Column(String(255))
    element_clicked = Column(String(100))
    properties = Column(JSONB)
    
    # Business impact
    is_conversion = Column(Boolean, default=False)
    conversion_value = Column(Numeric(10, 2))
    goal_achieved = Column(String(50))
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AgentPerformanceMetric(Base):
    """Agent performance metrics for business intelligence."""
    
    __tablename__ = "agent_performance_metrics"
    
    id = Column(DatabaseAgnosticUUID, primary_key=True, default=uuid.uuid4)
    agent_id = Column(DatabaseAgnosticUUID, ForeignKey("agents.id"), nullable=False, index=True)
    
    # Time period
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    measurement_period_minutes = Column(Integer, default=60)  # Measurement window
    
    # Performance metrics
    tasks_completed = Column(Integer, default=0)
    tasks_failed = Column(Integer, default=0)
    success_rate = Column(Numeric(5, 2))  # Percentage
    average_task_duration_seconds = Column(Integer)
    average_response_time_ms = Column(Integer)
    
    # Resource utilization
    cpu_usage_percent = Column(Numeric(5, 2))
    memory_usage_mb = Column(Integer)
    bandwidth_usage_mb = Column(Numeric(10, 3))
    api_calls_made = Column(Integer, default=0)
    
    # Business impact metrics
    user_satisfaction_score = Column(Numeric(3, 2))  # 1-5 scale
    business_value_generated = Column(Numeric(10, 2))
    cost_efficiency_score = Column(Numeric(5, 2))
    
    # Error and reliability metrics
    error_count = Column(Integer, default=0)
    uptime_percentage = Column(Numeric(5, 2))
    availability_percentage = Column(Numeric(5, 2))
    
    # Capacity metrics
    queue_depth = Column(Integer, default=0)
    utilization_percentage = Column(Numeric(5, 2))
    throughput_tasks_per_hour = Column(Integer)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class BusinessAlert(Base):
    """Business intelligence alerts and notifications."""
    
    __tablename__ = "business_alerts"
    
    id = Column(DatabaseAgnosticUUID, primary_key=True, default=uuid.uuid4)
    
    # Alert details
    alert_type = Column(String(50), nullable=False, index=True)
    alert_name = Column(String(100), nullable=False)
    alert_level = Column(SQLEnum(AlertLevel), nullable=False, index=True)
    message = Column(Text, nullable=False)
    description = Column(Text)
    
    # Alert triggers
    metric_name = Column(String(100), index=True)
    threshold_value = Column(Numeric(12, 4))
    current_value = Column(Numeric(12, 4))
    condition = Column(String(20))  # greater_than, less_than, equals, etc.
    
    # Alert timing
    triggered_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    acknowledged_at = Column(DateTime(timezone=True))
    resolved_at = Column(DateTime(timezone=True))
    
    # Alert context
    affected_components = Column(StringArray())
    related_metrics = Column(JSONB)
    suggested_actions = Column(StringArray())
    
    # Alert status
    is_active = Column(Boolean, default=True, index=True)
    is_acknowledged = Column(Boolean, default=False)
    is_resolved = Column(Boolean, default=False)
    
    # References
    agent_id = Column(DatabaseAgnosticUUID, ForeignKey("agents.id"), nullable=True, index=True)
    user_id = Column(DatabaseAgnosticUUID, ForeignKey("users.id"), nullable=True, index=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class BusinessForecast(Base):
    """Business forecasting and predictive analytics data."""
    
    __tablename__ = "business_forecasts"
    
    id = Column(DatabaseAgnosticUUID, primary_key=True, default=uuid.uuid4)
    
    # Forecast details
    forecast_type = Column(String(50), nullable=False, index=True)  # revenue, users, capacity
    forecast_name = Column(String(100), nullable=False)
    metric_name = Column(String(100), nullable=False)
    
    # Forecast period
    forecast_date = Column(DateTime(timezone=True), nullable=False, index=True)
    forecast_horizon_days = Column(Integer, nullable=False)  # How far ahead
    generated_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Forecast values
    predicted_value = Column(Numeric(12, 4), nullable=False)
    confidence_level = Column(Numeric(5, 2))  # 0-100 percentage
    lower_bound = Column(Numeric(12, 4))  # Confidence interval
    upper_bound = Column(Numeric(12, 4))  # Confidence interval
    
    # Model information
    model_name = Column(String(50))
    model_version = Column(String(20))
    model_accuracy = Column(Numeric(5, 2))  # Historical accuracy percentage
    training_data_points = Column(Integer)
    
    # Forecast metadata
    assumptions = Column(JSONB)
    influencing_factors = Column(StringArray())
    seasonality_adjusted = Column(Boolean, default=False)
    trend_adjusted = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class BusinessDashboardConfig(Base):
    """Configuration for business intelligence dashboards."""
    
    __tablename__ = "business_dashboard_configs"
    
    id = Column(DatabaseAgnosticUUID, primary_key=True, default=uuid.uuid4)
    
    # Dashboard details
    dashboard_name = Column(String(100), nullable=False, unique=True)
    dashboard_type = Column(String(50), nullable=False)  # executive, operational, analytical
    description = Column(Text)
    
    # Configuration
    widgets = Column(JSONB, nullable=False)  # Dashboard widget configuration
    layout = Column(JSONB)  # Layout configuration
    refresh_interval_seconds = Column(Integer, default=300)  # 5 minutes default
    
    # Access control
    is_public = Column(Boolean, default=False)
    allowed_roles = Column(StringArray())
    created_by = Column(DatabaseAgnosticUUID, ForeignKey("users.id"), nullable=True)
    
    # Metadata
    tags = Column(StringArray())
    version = Column(String(20), default="1.0")
    is_active = Column(Boolean, default=True, index=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# Add relationships to existing models if needed
# Note: These would be added to existing agent and user models
# AgentPerformanceMetric.agent = relationship("Agent", back_populates="performance_metrics")
# UserSession.user = relationship("User", back_populates="sessions")