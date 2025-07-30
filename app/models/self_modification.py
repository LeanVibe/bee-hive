"""
Self-Modification Engine Models

SQLAlchemy models for the self-modification engine that enables agents to 
safely evolve their code while maintaining system stability and security.
Supports code analysis, modification generation, sandbox testing, and rollback.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from decimal import Decimal
from enum import Enum as PyEnum

import sqlalchemy as sa
from sqlalchemy import Column, String, Text, Integer, Boolean, DateTime, JSON
from sqlalchemy.types import Numeric as SQLDecimal
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base
from app.core.database_types import DatabaseAgnosticUUID, UUIDArray, StringArray


class ModificationSafety(PyEnum):
    """Safety levels for code modifications."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class ModificationStatus(PyEnum):
    """Status of modification sessions."""
    ANALYZING = "analyzing"
    SUGGESTIONS_READY = "suggestions_ready"
    APPLYING = "applying"
    APPLIED = "applied"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    ARCHIVED = "archived"


class ModificationType(PyEnum):
    """Types of code modifications."""
    BUG_FIX = "bug_fix"
    PERFORMANCE = "performance"
    FEATURE_ADD = "feature_add"
    REFACTOR = "refactor"
    SECURITY_FIX = "security_fix"
    STYLE_IMPROVEMENT = "style_improvement"
    DEPENDENCY_UPDATE = "dependency_update"


class SandboxExecutionType(PyEnum):
    """Types of sandbox executions."""
    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    SECURITY_SCAN = "security_scan"
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    LINTING = "linting"
    TYPE_CHECK = "type_check"


class ModificationSession(Base):
    """
    Tracks self-modification analysis sessions and overall outcomes.
    
    Each session represents an agent's attempt to analyze and modify
    a codebase for specific goals (performance, bugs, features).
    """
    
    __tablename__ = "modification_sessions"
    
    id = Column(DatabaseAgnosticUUID(), primary_key=True, server_default=sa.text('gen_random_uuid()'))
    agent_id = Column(DatabaseAgnosticUUID(), sa.ForeignKey('agents.id', ondelete='CASCADE'), 
                      nullable=False, index=True)
    repository_id = Column(DatabaseAgnosticUUID(), sa.ForeignKey('github_repositories.id', ondelete='SET NULL'), 
                          nullable=True, index=True)
    
    # Session configuration
    codebase_path = Column(String(500), nullable=False, index=True)
    modification_goals = Column(JSON, nullable=False, server_default='[]')  # List of goals
    safety_level = Column(sa.Enum(ModificationSafety), nullable=False, 
                         server_default='conservative', index=True)
    status = Column(sa.Enum(ModificationStatus), nullable=False, 
                   server_default='analyzing', index=True)
    
    # Analysis context
    analysis_prompt = Column(Text, nullable=True)
    analysis_context = Column(JSON, nullable=True, server_default='{}')
    
    # Session metrics
    total_suggestions = Column(Integer, nullable=False, server_default='0')
    applied_modifications = Column(Integer, nullable=False, server_default='0')
    success_rate = Column(SQLDecimal(5,2), nullable=True)  # Percentage
    performance_improvement = Column(SQLDecimal(5,2), nullable=True)  # Percentage
    
    # Error tracking
    error_message = Column(Text, nullable=True)
    session_metadata = Column(JSON, nullable=True, server_default='{}')
    
    # Timestamps
    started_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    modifications = relationship("CodeModification", back_populates="session", cascade="all, delete-orphan")
    sandbox_executions = relationship("SandboxExecution", back_populates="session", cascade="all, delete-orphan")
    feedback = relationship("ModificationFeedback", back_populates="session", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<ModificationSession(id={self.id}, agent_id={self.agent_id}, status={self.status.value})>"
    
    @property
    def is_active(self) -> bool:
        """Check if session is currently active."""
        return self.status in [ModificationStatus.ANALYZING, ModificationStatus.SUGGESTIONS_READY, ModificationStatus.APPLYING]
    
    @property
    def duration_minutes(self) -> Optional[float]:
        """Calculate session duration in minutes."""
        if not self.completed_at:
            return None
        delta = self.completed_at - self.started_at
        return delta.total_seconds() / 60


class CodeModification(Base):
    """
    Individual code modifications with detailed tracking and safety scoring.
    
    Each modification represents a specific change to a file, with complete
    context about the change, safety assessment, and application status.
    """
    
    __tablename__ = "code_modifications"
    
    id = Column(DatabaseAgnosticUUID(), primary_key=True, server_default=sa.text('gen_random_uuid()'))
    session_id = Column(DatabaseAgnosticUUID(), sa.ForeignKey('modification_sessions.id', ondelete='CASCADE'), 
                        nullable=False, index=True)
    
    # File information
    file_path = Column(String(500), nullable=False, index=True)
    modification_type = Column(sa.Enum(ModificationType), nullable=False, index=True)
    
    # Content changes
    original_content = Column(Text, nullable=True)
    modified_content = Column(Text, nullable=True)
    content_diff = Column(Text, nullable=True)  # Unified diff
    
    # Reasoning and safety
    modification_reason = Column(Text, nullable=False)
    llm_reasoning = Column(Text, nullable=True)
    safety_score = Column(SQLDecimal(3,2), nullable=False)  # 0.0 to 1.0
    complexity_score = Column(SQLDecimal(3,2), nullable=True)  # 0.0 to 1.0
    performance_impact = Column(SQLDecimal(5,2), nullable=True)  # Expected percentage change
    
    # Change statistics
    lines_added = Column(Integer, nullable=True)
    lines_removed = Column(Integer, nullable=True)
    functions_modified = Column(Integer, nullable=True)
    dependencies_changed = Column(JSON, nullable=True, server_default='[]')
    test_files_affected = Column(JSON, nullable=True, server_default='[]')
    
    # Approval workflow
    approval_required = Column(Boolean, nullable=False, server_default='false')
    human_approved = Column(Boolean, nullable=True)
    approved_by = Column(String(255), nullable=True)
    approval_token = Column(String(500), nullable=True)
    
    # Git integration
    git_commit_hash = Column(String(40), nullable=True, index=True)
    git_branch = Column(String(255), nullable=True)
    rollback_commit_hash = Column(String(40), nullable=True)
    
    # Metadata
    modification_metadata = Column(JSON, nullable=True, server_default='{}')
    
    # Timestamps
    applied_at = Column(DateTime(timezone=True), nullable=True, index=True)
    rollback_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    session = relationship("ModificationSession", back_populates="modifications")
    metrics = relationship("ModificationMetric", back_populates="modification", cascade="all, delete-orphan")
    sandbox_executions = relationship("SandboxExecution", back_populates="modification", cascade="all, delete-orphan")
    feedback = relationship("ModificationFeedback", back_populates="modification", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<CodeModification(id={self.id}, file_path={self.file_path}, type={self.modification_type.value})>"
    
    @property
    def is_applied(self) -> bool:
        """Check if modification has been applied."""
        return self.applied_at is not None and self.rollback_at is None
    
    @property
    def is_rolled_back(self) -> bool:
        """Check if modification has been rolled back."""
        return self.rollback_at is not None
    
    @property
    def net_lines_changed(self) -> int:
        """Calculate net lines changed (added - removed)."""
        added = self.lines_added or 0
        removed = self.lines_removed or 0
        return added - removed
    
    @property
    def is_high_risk(self) -> bool:
        """Determine if modification is high risk based on safety score."""
        return self.safety_score < 0.5
    
    @property
    def requires_human_approval(self) -> bool:
        """Check if modification requires human approval."""
        return self.approval_required or self.is_high_risk


class ModificationMetric(Base):
    """
    Performance and quality metrics before/after modifications.
    
    Tracks various metrics to validate that modifications actually
    improve the codebase without introducing regressions.
    """
    
    __tablename__ = "modification_metrics"
    
    id = Column(DatabaseAgnosticUUID(), primary_key=True, server_default=sa.text('gen_random_uuid()'))
    modification_id = Column(DatabaseAgnosticUUID(), sa.ForeignKey('code_modifications.id', ondelete='CASCADE'), 
                            nullable=False, index=True)
    
    # Metric identification
    metric_name = Column(String(100), nullable=False, index=True)
    metric_category = Column(String(50), nullable=False, index=True)  # 'performance', 'quality', 'security'
    
    # Values
    baseline_value = Column(SQLDecimal(15,6), nullable=True)
    modified_value = Column(SQLDecimal(15,6), nullable=True)
    improvement_percentage = Column(SQLDecimal(8,4), nullable=True)
    
    # Measurement context
    measurement_unit = Column(String(50), nullable=True)  # 'ms', 'MB', 'percent', 'count'
    measurement_context = Column(String(200), nullable=True)  # Test case or scenario
    measurement_tool = Column(String(100), nullable=True)
    
    # Statistical data
    confidence_score = Column(SQLDecimal(3,2), nullable=True)  # 0.0 to 1.0
    statistical_significance = Column(Boolean, nullable=True)
    sample_size = Column(Integer, nullable=True)
    standard_deviation = Column(SQLDecimal(10,6), nullable=True)
    
    # Metadata
    measurement_metadata = Column(JSON, nullable=True, server_default='{}')
    
    # Timestamps
    measured_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    modification = relationship("CodeModification", back_populates="metrics")
    
    def __repr__(self) -> str:
        return f"<ModificationMetric(id={self.id}, metric_name={self.metric_name}, improvement={self.improvement_percentage}%)>"
    
    @property
    def is_improvement(self) -> bool:
        """Check if metric shows improvement."""
        return self.improvement_percentage is not None and self.improvement_percentage > 0
    
    @property
    def is_regression(self) -> bool:
        """Check if metric shows regression."""
        return self.improvement_percentage is not None and self.improvement_percentage < 0
    
    @property
    def is_significant_change(self, threshold: float = 5.0) -> bool:
        """Check if change is significant (default: 5% threshold)."""
        return abs(self.improvement_percentage or 0) >= threshold


class SandboxExecution(Base):
    """
    Results from isolated sandbox testing of modifications.
    
    Tracks all executions in isolated Docker containers to ensure
    modifications work correctly and don't introduce security issues.
    """
    
    __tablename__ = "sandbox_executions"
    
    id = Column(DatabaseAgnosticUUID(), primary_key=True, server_default=sa.text('gen_random_uuid()'))
    modification_id = Column(DatabaseAgnosticUUID(), sa.ForeignKey('code_modifications.id', ondelete='CASCADE'), 
                            nullable=False, index=True)
    session_id = Column(DatabaseAgnosticUUID(), sa.ForeignKey('modification_sessions.id', ondelete='CASCADE'), 
                       nullable=True, index=True)
    
    # Execution configuration
    execution_type = Column(sa.Enum(SandboxExecutionType), nullable=False, index=True)
    container_id = Column(String(100), nullable=True)
    image_name = Column(String(200), nullable=True)
    command = Column(Text, nullable=False)
    working_directory = Column(String(500), nullable=True)
    environment_variables = Column(JSON, nullable=True, server_default='{}')
    resource_limits = Column(JSON, nullable=True, server_default='{}')
    network_isolation = Column(Boolean, nullable=False, server_default='true')
    
    # Execution results
    stdout = Column(Text, nullable=True)
    stderr = Column(Text, nullable=True)
    exit_code = Column(Integer, nullable=True, index=True)
    
    # Resource usage
    execution_time_ms = Column(Integer, nullable=True)
    memory_usage_mb = Column(Integer, nullable=True)
    cpu_usage_percent = Column(SQLDecimal(5,2), nullable=True)
    disk_usage_mb = Column(Integer, nullable=True)
    
    # Security monitoring
    network_attempts = Column(Integer, nullable=False, server_default='0')
    security_violations = Column(JSON, nullable=True, server_default='[]')
    file_system_changes = Column(JSON, nullable=True, server_default='[]')
    
    # Results
    test_results = Column(JSON, nullable=True, server_default='{}')
    performance_metrics = Column(JSON, nullable=True, server_default='{}')
    sandbox_metadata = Column(JSON, nullable=True, server_default='{}')
    
    # Timestamps
    started_at = Column(DateTime(timezone=True), nullable=True, index=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    executed_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    modification = relationship("CodeModification", back_populates="sandbox_executions")
    session = relationship("ModificationSession", back_populates="sandbox_executions")
    
    def __repr__(self) -> str:
        return f"<SandboxExecution(id={self.id}, type={self.execution_type.value}, exit_code={self.exit_code})>"
    
    @property
    def is_successful(self) -> bool:
        """Check if execution was successful."""
        return self.exit_code == 0 and len(self.security_violations or []) == 0
    
    @property
    def has_security_violations(self) -> bool:
        """Check if execution had security violations."""
        return len(self.security_violations or []) > 0
    
    @property
    def execution_duration_seconds(self) -> Optional[float]:
        """Calculate execution duration in seconds."""
        if self.execution_time_ms is None:
            return None
        return self.execution_time_ms / 1000.0
    
    @property
    def is_timeout(self) -> bool:
        """Check if execution timed out (heuristic based on common timeout codes)."""
        return self.exit_code in [124, 128, 143]  # Common timeout exit codes


class ModificationFeedback(Base):
    """
    Learning and feedback tracking for context-aware improvements.
    
    Collects feedback from humans, automated systems, and metrics to
    improve future modification suggestions and learn project patterns.
    """
    
    __tablename__ = "modification_feedback"
    
    id = Column(DatabaseAgnosticUUID(), primary_key=True, server_default=sa.text('gen_random_uuid()'))
    modification_id = Column(DatabaseAgnosticUUID(), sa.ForeignKey('code_modifications.id', ondelete='CASCADE'), 
                            nullable=False, index=True)
    session_id = Column(DatabaseAgnosticUUID(), sa.ForeignKey('modification_sessions.id', ondelete='CASCADE'), 
                       nullable=True, index=True)
    
    # Feedback source and type
    feedback_source = Column(String(100), nullable=False, index=True)  # 'human', 'automated', 'metrics'
    feedback_type = Column(String(50), nullable=False, index=True)  # 'rating', 'comment', 'correction', 'approval'
    
    # Feedback content
    rating = Column(Integer, nullable=True)  # 1-5 scale
    feedback_text = Column(Text, nullable=True)
    
    # Learning data
    patterns_identified = Column(JSON, nullable=True, server_default='[]')
    anti_patterns_identified = Column(JSON, nullable=True, server_default='[]')
    improvement_suggestions = Column(JSON, nullable=True, server_default='[]')
    user_preferences = Column(JSON, nullable=True, server_default='{}')
    project_conventions = Column(JSON, nullable=True, server_default='{}')
    
    # Impact assessment
    impact_score = Column(SQLDecimal(3,2), nullable=True)  # 0.0 to 1.0
    applied_to_learning = Column(Boolean, nullable=False, server_default='false')
    
    # Metadata
    feedback_metadata = Column(JSON, nullable=True, server_default='{}')
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    modification = relationship("CodeModification", back_populates="feedback")
    session = relationship("ModificationSession", back_populates="feedback")
    
    def __repr__(self) -> str:
        return f"<ModificationFeedback(id={self.id}, source={self.feedback_source}, rating={self.rating})>"
    
    @property
    def is_positive_feedback(self) -> bool:
        """Check if feedback is positive (rating >= 4)."""
        return self.rating is not None and self.rating >= 4
    
    @property
    def is_negative_feedback(self) -> bool:
        """Check if feedback is negative (rating <= 2)."""
        return self.rating is not None and self.rating <= 2
    
    @property
    def has_learning_data(self) -> bool:
        """Check if feedback contains learning data."""
        return (
            len(self.patterns_identified or []) > 0 or
            len(self.anti_patterns_identified or []) > 0 or
            len(self.improvement_suggestions or []) > 0
        )


# Export all models for easy import
__all__ = [
    "ModificationSafety",
    "ModificationStatus", 
    "ModificationType",
    "SandboxExecutionType",
    "ModificationSession",
    "CodeModification",
    "ModificationMetric",
    "SandboxExecution",
    "ModificationFeedback",
]