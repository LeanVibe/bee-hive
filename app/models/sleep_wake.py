"""
Enhanced Sleep-Wake Manager models for autonomous consolidation and recovery.
"""

import uuid
from datetime import datetime, time, date
from typing import Dict, Any, List, Optional
from enum import Enum

from sqlalchemy import (
    Column, DateTime, JSON, Enum as SQLEnum, Text, ForeignKey, 
    String, Boolean, Integer, Float, BigInteger, Time, Date, Index
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from ..core.database import Base


class SleepState(Enum):
    """Current sleep state of an agent."""
    AWAKE = "AWAKE"
    PREPARING_SLEEP = "PREPARING_SLEEP"
    SLEEPING = "SLEEPING"
    CONSOLIDATING = "CONSOLIDATING"
    PREPARING_WAKE = "PREPARING_WAKE"
    ERROR = "ERROR"


class CheckpointType(Enum):
    """Types of system checkpoints."""
    SCHEDULED = "SCHEDULED"
    PRE_SLEEP = "PRE_SLEEP"
    ERROR_RECOVERY = "ERROR_RECOVERY"
    MANUAL = "MANUAL"
    EMERGENCY = "EMERGENCY"


class ConsolidationStatus(Enum):
    """Status of consolidation jobs."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class SleepWindow(Base):
    """Sleep window configuration for agent scheduling."""
    
    __tablename__ = "sleep_windows"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id", ondelete="CASCADE"), nullable=True, index=True)
    start_time = Column(Time, nullable=False)
    end_time = Column(Time, nullable=False)
    timezone = Column(String(64), nullable=False, default="UTC")
    active = Column(Boolean, nullable=False, default=True)
    days_of_week = Column(JSON, nullable=False, default=lambda: [1, 2, 3, 4, 5, 6, 7])  # 1=Monday, 7=Sunday
    priority = Column(Integer, nullable=False, default=0)  # Higher priority overrides lower
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    agent = relationship("Agent", back_populates="sleep_windows")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": str(self.agent_id) if self.agent_id else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "timezone": self.timezone,
            "active": self.active,
            "days_of_week": self.days_of_week,
            "priority": self.priority,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    def is_time_in_window(self, check_time: time) -> bool:
        """Check if given time falls within this sleep window."""
        if self.start_time <= self.end_time:
            # Same day window (e.g., 02:00 - 04:00)
            return self.start_time <= check_time <= self.end_time
        else:
            # Overnight window (e.g., 23:00 - 02:00)
            return check_time >= self.start_time or check_time <= self.end_time


class Checkpoint(Base):
    """System checkpoints for atomic state preservation."""
    
    __tablename__ = "checkpoints"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id", ondelete="CASCADE"), nullable=True, index=True)
    checkpoint_type = Column(SQLEnum(CheckpointType), nullable=False, index=True)
    path = Column(Text, nullable=False)
    sha256 = Column(String(64), nullable=False)
    size_bytes = Column(BigInteger, nullable=False)
    is_valid = Column(Boolean, nullable=False, default=False)
    validation_errors = Column(JSON, nullable=True, default=list)
    checkpoint_metadata = Column(JSON, nullable=True, default=dict)
    redis_offsets = Column(JSON, nullable=True, default=dict)  # Stream offsets snapshot
    database_snapshot_id = Column(String(255), nullable=True)  # Git commit or database backup ID
    compression_ratio = Column(Float, nullable=True)
    creation_time_ms = Column(Float, nullable=True)
    validation_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)  # For cleanup policies
    
    # Relationships
    agent = relationship("Agent", back_populates="checkpoints")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "agent_id": str(self.agent_id) if self.agent_id else None,
            "checkpoint_type": self.checkpoint_type.value,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
            "metadata": self.checkpoint_metadata,
            "redis_offsets": self.redis_offsets,
            "database_snapshot_id": self.database_snapshot_id,
            "compression_ratio": self.compression_ratio,
            "creation_time_ms": self.creation_time_ms,
            "validation_time_ms": self.validation_time_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }
    
    @property
    def size_mb(self) -> float:
        """Return size in megabytes."""
        return self.size_bytes / (1024 * 1024)
    
    def validate_integrity(self) -> bool:
        """Validate checkpoint integrity using SHA-256."""
        # This would be implemented to actually verify the file checksum
        # For now, return the stored validation state
        return self.is_valid


class SleepWakeCycle(Base):
    """Enhanced sleep-wake cycle tracking with detailed state management."""
    
    __tablename__ = "sleep_wake_cycles"
    __table_args__ = {'extend_existing': True}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id", ondelete="CASCADE"), nullable=False, index=True)
    cycle_type = Column(String(50), nullable=False, index=True)  # Keep flexible for new types
    sleep_state = Column(SQLEnum(SleepState), nullable=False, default=SleepState.AWAKE, index=True)
    sleep_time = Column(DateTime(timezone=True), nullable=True)
    wake_time = Column(DateTime(timezone=True), nullable=True)
    expected_wake_time = Column(DateTime(timezone=True), nullable=True)
    pre_sleep_checkpoint_id = Column(UUID(as_uuid=True), ForeignKey("checkpoints.id", ondelete="SET NULL"), nullable=True)
    post_wake_checkpoint_id = Column(UUID(as_uuid=True), ForeignKey("checkpoints.id", ondelete="SET NULL"), nullable=True)
    consolidation_summary = Column(Text, nullable=True)
    context_changes = Column(JSON, nullable=True, default=dict)
    performance_metrics = Column(JSON, nullable=True, default=dict)
    error_details = Column(JSON, nullable=True, default=dict)
    token_reduction_achieved = Column(Float, nullable=True)
    consolidation_time_ms = Column(Float, nullable=True)
    recovery_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    agent = relationship("Agent", back_populates="sleep_wake_cycles")
    pre_sleep_checkpoint = relationship("Checkpoint", foreign_keys=[pre_sleep_checkpoint_id])
    post_wake_checkpoint = relationship("Checkpoint", foreign_keys=[post_wake_checkpoint_id])
    consolidation_jobs = relationship("ConsolidationJob", back_populates="cycle")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "agent_id": str(self.agent_id),
            "cycle_type": self.cycle_type,
            "sleep_state": self.sleep_state.value,
            "sleep_time": self.sleep_time.isoformat() if self.sleep_time else None,
            "wake_time": self.wake_time.isoformat() if self.wake_time else None,
            "expected_wake_time": self.expected_wake_time.isoformat() if self.expected_wake_time else None,
            "pre_sleep_checkpoint_id": str(self.pre_sleep_checkpoint_id) if self.pre_sleep_checkpoint_id else None,
            "post_wake_checkpoint_id": str(self.post_wake_checkpoint_id) if self.post_wake_checkpoint_id else None,
            "consolidation_summary": self.consolidation_summary,
            "context_changes": self.context_changes,
            "performance_metrics": self.performance_metrics,
            "error_details": self.error_details,
            "token_reduction_achieved": self.token_reduction_achieved,
            "consolidation_time_ms": self.consolidation_time_ms,
            "recovery_time_ms": self.recovery_time_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    @property
    def duration_minutes(self) -> Optional[float]:
        """Calculate cycle duration in minutes."""
        if self.sleep_time and self.wake_time:
            delta = self.wake_time - self.sleep_time
            return delta.total_seconds() / 60
        return None
    
    @property
    def is_active(self) -> bool:
        """Check if cycle is currently active."""
        return self.sleep_state in [SleepState.SLEEPING, SleepState.CONSOLIDATING, SleepState.PREPARING_WAKE]
    
    def get_efficiency_score(self) -> Optional[float]:
        """Calculate efficiency score based on token reduction and time."""
        if self.token_reduction_achieved and self.consolidation_time_ms:
            # Score based on tokens saved per second
            return self.token_reduction_achieved / (self.consolidation_time_ms / 1000)
        return None


class ConsolidationJob(Base):
    """Consolidation jobs for tracking background work during sleep cycles."""
    
    __tablename__ = "consolidation_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    cycle_id = Column(UUID(as_uuid=True), ForeignKey("sleep_wake_cycles.id", ondelete="CASCADE"), nullable=False, index=True)
    job_type = Column(String(50), nullable=False, index=True)  # context_compression, vector_update, etc.
    status = Column(SQLEnum(ConsolidationStatus), nullable=False, default=ConsolidationStatus.PENDING, index=True)
    input_data = Column(JSON, nullable=True, default=dict)
    output_data = Column(JSON, nullable=True, default=dict)
    error_message = Column(Text, nullable=True)
    progress_percentage = Column(Float, nullable=False, default=0.0)
    processing_time_ms = Column(Float, nullable=True)
    tokens_processed = Column(Integer, nullable=True)
    tokens_saved = Column(Integer, nullable=True)
    priority = Column(Integer, nullable=False, default=0)
    retry_count = Column(Integer, nullable=False, default=0)
    max_retries = Column(Integer, nullable=False, default=3)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    cycle = relationship("SleepWakeCycle", back_populates="consolidation_jobs")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "cycle_id": str(self.cycle_id),
            "job_type": self.job_type,
            "status": self.status.value,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error_message": self.error_message,
            "progress_percentage": self.progress_percentage,
            "processing_time_ms": self.processing_time_ms,
            "tokens_processed": self.tokens_processed,
            "tokens_saved": self.tokens_saved,
            "priority": self.priority,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
    
    @property
    def can_retry(self) -> bool:
        """Check if job can be retried."""
        return self.retry_count < self.max_retries and self.status == ConsolidationStatus.FAILED
    
    @property
    def efficiency_ratio(self) -> Optional[float]:
        """Calculate efficiency ratio (tokens saved / tokens processed)."""
        if self.tokens_processed and self.tokens_processed > 0:
            return (self.tokens_saved or 0) / self.tokens_processed
        return None


class SleepWakeAnalytics(Base):
    """Sleep-wake analytics for performance tracking and optimization."""
    
    __tablename__ = "sleep_wake_analytics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id", ondelete="CASCADE"), nullable=True, index=True)
    date = Column(Date, nullable=False, index=True)
    total_cycles = Column(Integer, nullable=False, default=0)
    successful_cycles = Column(Integer, nullable=False, default=0)
    failed_cycles = Column(Integer, nullable=False, default=0)
    average_token_reduction = Column(Float, nullable=True)
    average_consolidation_time_ms = Column(Float, nullable=True)
    average_recovery_time_ms = Column(Float, nullable=True)
    total_tokens_saved = Column(BigInteger, nullable=False, default=0)
    total_processing_time_ms = Column(Float, nullable=False, default=0.0)
    uptime_percentage = Column(Float, nullable=True)
    checkpoints_created = Column(Integer, nullable=False, default=0)
    checkpoints_validated = Column(Integer, nullable=False, default=0)
    fallback_recoveries = Column(Integer, nullable=False, default=0)
    manual_interventions = Column(Integer, nullable=False, default=0)
    updated_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    agent = relationship("Agent", back_populates="sleep_wake_analytics")
    
    __table_args__ = (
        Index('idx_sleep_wake_analytics_date', 'date'),
        Index('idx_sleep_wake_analytics_agent_date', 'agent_id', 'date'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "agent_id": str(self.agent_id) if self.agent_id else None,
            "date": self.date.isoformat() if self.date else None,
            "total_cycles": self.total_cycles,
            "successful_cycles": self.successful_cycles,
            "failed_cycles": self.failed_cycles,
            "average_token_reduction": self.average_token_reduction,
            "average_consolidation_time_ms": self.average_consolidation_time_ms,
            "average_recovery_time_ms": self.average_recovery_time_ms,
            "total_tokens_saved": self.total_tokens_saved,
            "total_processing_time_ms": self.total_processing_time_ms,
            "uptime_percentage": self.uptime_percentage,
            "checkpoints_created": self.checkpoints_created,
            "checkpoints_validated": self.checkpoints_validated,
            "fallback_recoveries": self.fallback_recoveries,
            "manual_interventions": self.manual_interventions,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_cycles > 0:
            return (self.successful_cycles / self.total_cycles) * 100
        return 0.0
    
    @property
    def average_tokens_saved_per_cycle(self) -> float:
        """Calculate average tokens saved per cycle."""
        if self.successful_cycles > 0:
            return self.total_tokens_saved / self.successful_cycles
        return 0.0


# Legacy support - keep the old model for backward compatibility
class CycleType(Enum):
    """Types of sleep-wake cycles."""
    LIGHT_CONSOLIDATION = "light_consolidation"
    DEEP_SLEEP = "deep_sleep"
    EMERGENCY_CONSOLIDATION = "emergency_consolidation"