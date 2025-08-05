"""
Alert models for intelligence system.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Float, Boolean
from sqlalchemy.sql import func

from app.core.database import Base


class Alert(Base):
    """Alert model for storing alert data and analysis"""
    __tablename__ = "alerts"
    
    id = Column(String, primary_key=True)
    alert_type = Column(String, nullable=False, index=True)
    severity = Column(String, nullable=False, index=True)
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=func.now(), index=True)
    alert_metadata = Column(JSON, default={})
    
    # Analysis results
    analysis_results = Column(JSON, default={})
    priority_score = Column(Float, default=0.5)
    pattern_ids = Column(JSON, default=[])
    
    # Tracking fields
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)
    response_time = Column(Float, nullable=True)  # seconds
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class AlertPattern(Base):
    """Alert pattern model for storing detected patterns"""
    __tablename__ = "alert_patterns"
    
    id = Column(String, primary_key=True)
    pattern_type = Column(String, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    frequency = Column(Integer, default=1)
    
    first_seen = Column(DateTime, nullable=False)
    last_seen = Column(DateTime, nullable=False)
    
    related_alerts = Column(JSON, default=[])
    pattern_metadata = Column(JSON, default={})
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class AlertFrequency(Base):
    """Alert frequency tracking for pattern analysis"""
    __tablename__ = "alert_frequencies"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_type = Column(String, nullable=False, index=True)
    hour_bucket = Column(DateTime, nullable=False, index=True)  # Rounded to hour
    count = Column(Integer, default=1)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())