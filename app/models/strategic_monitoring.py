"""
Strategic Monitoring Database Models

Database models for storing strategic monitoring and analytics data including:
- Market intelligence and competitive analysis data
- Performance metrics and KPI tracking
- Strategic recommendations and risk assessments
- Intelligence sources and monitoring configurations
- Real-time alerting and notification data

Designed to support comprehensive strategic monitoring for global market expansion.
"""

from datetime import datetime, date
from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4
from enum import Enum as PyEnum

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Date, Text, JSON,
    ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID, ARRAY
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func

from ..core.database import Base


class MarketSegmentType(str, PyEnum):
    """Market segment types."""
    ENTERPRISE = "enterprise"
    SMB = "smb"
    STARTUP = "startup"
    INDIVIDUAL = "individual"
    GOVERNMENT = "government"
    ACADEMIA = "academia"


class CompetitivePositionType(str, PyEnum):
    """Competitive position types."""
    LEADER = "leader"
    CHALLENGER = "challenger"
    VISIONARY = "visionary"
    NICHE_PLAYER = "niche_player"
    EMERGING = "emerging"


class RiskLevelType(str, PyEnum):
    """Risk level types."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConfidenceLevelType(str, PyEnum):
    """Confidence level types."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


class AlertSeverityType(str, PyEnum):
    """Alert severity types."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class MarketIntelligenceData(Base):
    """Market intelligence data storage."""
    __tablename__ = "market_intelligence_data"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    analysis_id = Column(String(255), nullable=False, index=True)
    market_segment = Column(String(50), nullable=False)
    region = Column(String(100), nullable=False)
    market_size_usd = Column(Float, nullable=False)
    growth_rate_percent = Column(Float, nullable=False)
    key_players = Column(JSON, nullable=False, default=list)
    adoption_rate_percent = Column(Float, nullable=False)
    sentiment_score = Column(Float, nullable=False)  # -1.0 to 1.0
    confidence_level = Column(String(20), nullable=False)
    data_sources = Column(JSON, nullable=False, default=list)
    extra_metadata = Column(JSON, nullable=True, default=dict)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    __table_args__ = (
        Index('ix_market_intelligence_segment_region', 'market_segment', 'region'),
        Index('ix_market_intelligence_created_at', 'created_at'),
        CheckConstraint('sentiment_score >= -1.0 AND sentiment_score <= 1.0', name='sentiment_score_range'),
        CheckConstraint('adoption_rate_percent >= 0.0 AND adoption_rate_percent <= 100.0', name='adoption_rate_range'),
    )


class CompetitorAnalysisData(Base):
    """Competitor analysis data storage."""
    __tablename__ = "competitor_analysis_data"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    analysis_id = Column(String(255), nullable=False, index=True)
    competitor_name = Column(String(255), nullable=False, index=True)
    market_position = Column(String(50), nullable=False)
    market_share_percent = Column(Float, nullable=False)
    strengths = Column(JSON, nullable=False, default=list)
    weaknesses = Column(JSON, nullable=False, default=list)
    threat_level = Column(String(20), nullable=False)
    strategic_moves = Column(JSON, nullable=False, default=list)
    financial_health_score = Column(Float, nullable=False)  # 0.0 to 100.0
    innovation_score = Column(Float, nullable=False)  # 0.0 to 100.0
    customer_satisfaction_score = Column(Float, nullable=False)  # 0.0 to 100.0
    data_sources = Column(JSON, nullable=False, default=list)
    extra_metadata = Column(JSON, nullable=True, default=dict)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    __table_args__ = (
        Index('ix_competitor_analysis_competitor', 'competitor_name'),
        Index('ix_competitor_analysis_created_at', 'created_at'),
        CheckConstraint('market_share_percent >= 0.0 AND market_share_percent <= 100.0', name='market_share_range'),
        CheckConstraint('financial_health_score >= 0.0 AND financial_health_score <= 100.0', name='financial_health_range'),
        CheckConstraint('innovation_score >= 0.0 AND innovation_score <= 100.0', name='innovation_score_range'),
        CheckConstraint('customer_satisfaction_score >= 0.0 AND customer_satisfaction_score <= 100.0', name='customer_satisfaction_range'),
    )


class StrategicPerformanceMetrics(Base):
    """Strategic performance metrics tracking."""
    __tablename__ = "strategic_performance_metrics"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    metric_id = Column(String(255), nullable=False, index=True)
    metric_name = Column(String(255), nullable=False)
    category = Column(String(100), nullable=False)
    current_value = Column(Float, nullable=False)
    target_value = Column(Float, nullable=False)
    unit = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False)
    trend_direction = Column(String(50), nullable=False)
    change_percent = Column(Float, nullable=False)
    confidence_level = Column(Float, nullable=False)  # 0.0 to 1.0
    data_sources = Column(JSON, nullable=False, default=list)
    extra_metadata = Column(JSON, nullable=True, default=dict)
    reporting_period = Column(String(50), nullable=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationship to historical data
    historical_data = relationship("StrategicPerformanceMetricHistory", back_populates="metric", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('ix_performance_metrics_category', 'category'),
        Index('ix_performance_metrics_name', 'metric_name'),
        Index('ix_performance_metrics_period', 'reporting_period'),
        UniqueConstraint('metric_id', 'reporting_period', name='uq_metric_period'),
        CheckConstraint('confidence_level >= 0.0 AND confidence_level <= 1.0', name='confidence_level_range'),
    )


class StrategicPerformanceMetricHistory(Base):
    """Historical strategic performance metrics data."""
    __tablename__ = "strategic_performance_metric_history"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    metric_id = Column(PGUUID(as_uuid=True), ForeignKey('strategic_performance_metrics.id'), nullable=False)
    recorded_at = Column(DateTime(timezone=True), nullable=False)
    value = Column(Float, nullable=False)
    extra_metadata = Column(JSON, nullable=True, default=dict)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship
    metric = relationship("StrategicPerformanceMetrics", back_populates="historical_data")
    
    __table_args__ = (
        Index('ix_metric_history_metric_recorded', 'metric_id', 'recorded_at'),
        Index('ix_metric_history_recorded_at', 'recorded_at'),
    )


class StrategicRecommendations(Base):
    """Strategic recommendations storage."""
    __tablename__ = "strategic_recommendations"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    recommendation_id = Column(String(255), nullable=False, unique=True, index=True)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    category = Column(String(100), nullable=False)
    priority_score = Column(Float, nullable=False)  # 0.0 to 100.0
    confidence_level = Column(String(20), nullable=False)
    investment_required_usd = Column(Float, nullable=True)
    expected_roi_percent = Column(Float, nullable=True)
    time_to_impact_months = Column(Integer, nullable=False)
    risk_assessment = Column(String(20), nullable=False)
    success_metrics = Column(JSON, nullable=False, default=list)
    action_items = Column(JSON, nullable=False, default=list)
    stakeholders = Column(JSON, nullable=False, default=list)
    supporting_intelligence = Column(JSON, nullable=False, default=list)
    status = Column(String(50), nullable=False, default='pending')
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    __table_args__ = (
        Index('ix_strategic_recommendations_category', 'category'),
        Index('ix_strategic_recommendations_priority', 'priority_score'),
        Index('ix_strategic_recommendations_status', 'status'),
        CheckConstraint('priority_score >= 0.0 AND priority_score <= 100.0', name='priority_score_range'),
        CheckConstraint('time_to_impact_months > 0', name='time_to_impact_positive'),
    )


class RiskAssessments(Base):
    """Risk assessment data storage."""
    __tablename__ = "risk_assessments"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    assessment_id = Column(String(255), nullable=False, unique=True, index=True)
    risk_category = Column(String(100), nullable=False)
    risk_description = Column(Text, nullable=False)
    probability = Column(Float, nullable=False)  # 0.0 to 1.0
    impact_score = Column(Float, nullable=False)  # 0.0 to 100.0
    risk_level = Column(String(20), nullable=False)
    time_horizon = Column(String(50), nullable=False)
    mitigation_strategies = Column(JSON, nullable=False, default=list)
    monitoring_indicators = Column(JSON, nullable=False, default=list)
    responsible_stakeholders = Column(JSON, nullable=False, default=list)
    trend_direction = Column(String(50), nullable=False)
    related_risks = Column(JSON, nullable=False, default=list)
    status = Column(String(50), nullable=False, default='active')
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    __table_args__ = (
        Index('ix_risk_assessments_category', 'risk_category'),
        Index('ix_risk_assessments_level', 'risk_level'),
        Index('ix_risk_assessments_status', 'status'),
        CheckConstraint('probability >= 0.0 AND probability <= 1.0', name='probability_range'),
        CheckConstraint('impact_score >= 0.0 AND impact_score <= 100.0', name='impact_score_range'),
    )


class IntelligenceAlerts(Base):
    """Intelligence-driven alerts storage."""
    __tablename__ = "intelligence_alerts"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    alert_id = Column(String(255), nullable=False, unique=True, index=True)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    severity = Column(String(20), nullable=False)
    intelligence_type = Column(String(50), nullable=False)
    trigger_conditions = Column(JSON, nullable=False, default=dict)
    affected_areas = Column(JSON, nullable=False, default=list)
    recommended_actions = Column(JSON, nullable=False, default=list)
    escalation_path = Column(JSON, nullable=False, default=list)
    acknowledged = Column(Boolean, nullable=False, default=False)
    acknowledged_by = Column(String(255), nullable=True)
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    resolved = Column(Boolean, nullable=False, default=False)
    resolved_by = Column(String(255), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    resolution_notes = Column(Text, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    __table_args__ = (
        Index('ix_intelligence_alerts_severity', 'severity'),
        Index('ix_intelligence_alerts_type', 'intelligence_type'),
        Index('ix_intelligence_alerts_status', 'acknowledged', 'resolved'),
        Index('ix_intelligence_alerts_created_at', 'created_at'),
    )


class IntelligenceMonitors(Base):
    """Intelligence monitoring configurations."""
    __tablename__ = "intelligence_monitors"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    monitor_id = Column(String(255), nullable=False, unique=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    monitor_type = Column(String(100), nullable=False)
    configuration = Column(JSON, nullable=False, default=dict)
    alert_triggers = Column(JSON, nullable=False, default=dict)
    data_sources = Column(JSON, nullable=False, default=list)
    status = Column(String(50), nullable=False, default='active')
    execution_frequency = Column(String(50), nullable=False)
    last_execution = Column(DateTime(timezone=True), nullable=True)
    next_execution = Column(DateTime(timezone=True), nullable=True)
    execution_count = Column(Integer, nullable=False, default=0)
    success_count = Column(Integer, nullable=False, default=0)
    failure_count = Column(Integer, nullable=False, default=0)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    __table_args__ = (
        Index('ix_intelligence_monitors_type', 'monitor_type'),
        Index('ix_intelligence_monitors_status', 'status'),
        Index('ix_intelligence_monitors_next_execution', 'next_execution'),
    )


class MarketTrends(Base):
    """Market trends analysis data."""
    __tablename__ = "market_trends"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    trend_id = Column(String(255), nullable=False, unique=True, index=True)
    trend_name = Column(String(255), nullable=False)
    category = Column(String(100), nullable=False)
    direction = Column(String(50), nullable=False)
    impact_score = Column(Float, nullable=False)  # 0.0 to 100.0
    velocity_score = Column(Float, nullable=False)  # Rate of change
    time_horizon_months = Column(Integer, nullable=False)
    affected_segments = Column(JSON, nullable=False, default=list)
    opportunities = Column(JSON, nullable=False, default=list)
    threats = Column(JSON, nullable=False, default=list)
    confidence_level = Column(Float, nullable=False)  # 0.0 to 1.0
    supporting_data = Column(JSON, nullable=False, default=dict)
    
    first_detected = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    __table_args__ = (
        Index('ix_market_trends_category', 'category'),
        Index('ix_market_trends_direction', 'direction'),
        Index('ix_market_trends_impact', 'impact_score'),
        Index('ix_market_trends_detected', 'first_detected'),
        CheckConstraint('impact_score >= 0.0 AND impact_score <= 100.0', name='impact_score_range'),
        CheckConstraint('confidence_level >= 0.0 AND confidence_level <= 1.0', name='confidence_level_range'),
        CheckConstraint('time_horizon_months > 0', name='time_horizon_positive'),
    )


class BusinessMetrics(Base):
    """Business performance metrics."""
    __tablename__ = "business_metrics"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    metric_date = Column(Date, nullable=False)
    metric_type = Column(String(100), nullable=False)
    
    # Enterprise Partnership Metrics
    total_partnerships = Column(Integer, nullable=True)
    active_partnerships = Column(Integer, nullable=True)
    pipeline_value_usd = Column(Float, nullable=True)
    partnership_conversion_rate = Column(Float, nullable=True)
    average_deal_size_usd = Column(Float, nullable=True)
    partnership_revenue_usd = Column(Float, nullable=True)
    
    # Community Growth Metrics
    total_community_members = Column(Integer, nullable=True)
    monthly_active_users = Column(Integer, nullable=True)
    daily_active_users = Column(Integer, nullable=True)
    community_growth_rate = Column(Float, nullable=True)
    engagement_rate = Column(Float, nullable=True)
    viral_coefficient = Column(Float, nullable=True)
    
    # Thought Leadership Metrics
    content_pieces_published = Column(Integer, nullable=True)
    total_content_views = Column(Integer, nullable=True)
    influence_score = Column(Float, nullable=True)
    brand_sentiment_score = Column(Float, nullable=True)
    speaking_engagements = Column(Integer, nullable=True)
    
    # Revenue Metrics
    total_revenue_usd = Column(Float, nullable=True)
    recurring_revenue_usd = Column(Float, nullable=True)
    revenue_growth_rate = Column(Float, nullable=True)
    customer_acquisition_cost = Column(Float, nullable=True)
    customer_lifetime_value = Column(Float, nullable=True)
    
    extra_metadata = Column(JSON, nullable=True, default=dict)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    __table_args__ = (
        Index('ix_business_metrics_date_type', 'metric_date', 'metric_type'),
        Index('ix_business_metrics_date', 'metric_date'),
        Index('ix_business_metrics_type', 'metric_type'),
        UniqueConstraint('metric_date', 'metric_type', name='uq_business_metrics_date_type'),
    )


class StrategicOpportunities(Base):
    """Strategic market opportunities."""
    __tablename__ = "strategic_opportunities"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    opportunity_id = Column(String(255), nullable=False, unique=True, index=True)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    market_segment = Column(String(50), nullable=False)
    region = Column(String(100), nullable=False)
    size_estimate_usd = Column(Float, nullable=False)
    confidence_level = Column(Float, nullable=False)  # 0.0 to 1.0
    competition_level = Column(String(20), nullable=False)
    barriers_to_entry = Column(JSON, nullable=False, default=list)
    key_success_factors = Column(JSON, nullable=False, default=list)
    timeline_months = Column(Integer, nullable=False)
    supporting_trends = Column(JSON, nullable=False, default=list)
    priority_score = Column(Float, nullable=False)  # 0.0 to 100.0
    status = Column(String(50), nullable=False, default='identified')
    
    identified_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    __table_args__ = (
        Index('ix_strategic_opportunities_segment', 'market_segment'),
        Index('ix_strategic_opportunities_region', 'region'),
        Index('ix_strategic_opportunities_priority', 'priority_score'),
        Index('ix_strategic_opportunities_status', 'status'),
        CheckConstraint('confidence_level >= 0.0 AND confidence_level <= 1.0', name='confidence_level_range'),
        CheckConstraint('priority_score >= 0.0 AND priority_score <= 100.0', name='priority_score_range'),
        CheckConstraint('timeline_months > 0', name='timeline_positive'),
    )