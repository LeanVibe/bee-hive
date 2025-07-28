"""
SQLAlchemy models for Prompt Optimization System.

Defines database models for prompt templates, optimization experiments,
variants, evaluations, A/B tests, feedback, and test cases.
"""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from sqlalchemy import (
    Column, String, Text, Integer, Float, Boolean, DateTime, JSON,
    ForeignKey, CheckConstraint, Index
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

from ..core.database import Base


class PromptStatus(str, Enum):
    """Status of a prompt template."""
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class ExperimentStatus(str, Enum):
    """Status of an optimization experiment."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OptimizationMethod(str, Enum):
    """Methods for prompt optimization."""
    META_PROMPTING = "meta_prompting"
    EVOLUTIONARY = "evolutionary"
    GRADIENT_BASED = "gradient_based"
    FEW_SHOT = "few_shot"
    MANUAL = "manual"


class PromptTemplate(Base):
    """Base prompt template for optimization."""
    
    __tablename__ = "prompt_templates"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4,
        server_default=func.gen_random_uuid()
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    task_type: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    domain: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    template_content: Mapped[str] = mapped_column(Text, nullable=False)
    template_variables: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    version: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    status: Mapped[PromptStatus] = mapped_column(
        String(20), 
        default=PromptStatus.DRAFT, 
        server_default="draft"
    )
    created_by: Mapped[Optional[str]] = mapped_column(String(255))
    description: Mapped[Optional[str]] = mapped_column(Text)
    tags: Mapped[List[str]] = mapped_column(JSON, default=list)
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=datetime.utcnow,
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        server_default=func.now()
    )
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Relationships
    experiments: Mapped[List["OptimizationExperiment"]] = relationship(
        "OptimizationExperiment", 
        back_populates="base_prompt",
        cascade="all, delete-orphan"
    )
    variants: Mapped[List["PromptVariant"]] = relationship(
        "PromptVariant", 
        back_populates="parent_prompt",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<PromptTemplate(id={self.id}, name='{self.name}', status='{self.status}')>"


class OptimizationExperiment(Base):
    """Optimization experiment for a prompt template."""
    
    __tablename__ = "optimization_experiments"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4,
        server_default=func.gen_random_uuid()
    )
    experiment_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    base_prompt_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("prompt_templates.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    optimization_method: Mapped[OptimizationMethod] = mapped_column(String(50), nullable=False)
    target_metrics: Mapped[Dict[str, float]] = mapped_column(JSON, default=dict)
    experiment_config: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[ExperimentStatus] = mapped_column(
        String(20), 
        default=ExperimentStatus.PENDING,
        server_default="pending"
    )
    
    # Progress tracking
    progress_percentage: Mapped[float] = mapped_column(Float, default=0.0, server_default="0.0")
    current_iteration: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    max_iterations: Mapped[int] = mapped_column(Integer, default=50, server_default="50")
    best_score: Mapped[Optional[float]] = mapped_column(Float)
    baseline_score: Mapped[Optional[float]] = mapped_column(Float)
    improvement_percentage: Mapped[Optional[float]] = mapped_column(Float)
    convergence_threshold: Mapped[float] = mapped_column(Float, default=0.01, server_default="0.01")
    early_stopping: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    
    # Metadata
    created_by: Mapped[Optional[str]] = mapped_column(String(255))
    description: Mapped[Optional[str]] = mapped_column(Text)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    
    # Timestamps
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=datetime.utcnow,
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        server_default=func.now()
    )
    
    # Relationships
    base_prompt: Mapped["PromptTemplate"] = relationship(
        "PromptTemplate", 
        back_populates="experiments"
    )
    variants: Mapped[List["PromptVariant"]] = relationship(
        "PromptVariant", 
        back_populates="experiment",
        cascade="all, delete-orphan"
    )
    ab_tests: Mapped[List["ABTestResult"]] = relationship(
        "ABTestResult", 
        back_populates="experiment",
        cascade="all, delete-orphan"
    )
    metrics: Mapped[List["OptimizationMetric"]] = relationship(
        "OptimizationMetric", 
        back_populates="experiment",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<OptimizationExperiment(id={self.id}, name='{self.experiment_name}', status='{self.status}')>"


class PromptVariant(Base):
    """Generated prompt variant from optimization."""
    
    __tablename__ = "prompt_variants"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4,
        server_default=func.gen_random_uuid()
    )
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("optimization_experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    parent_prompt_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("prompt_templates.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    variant_content: Mapped[str] = mapped_column(Text, nullable=False)
    generation_method: Mapped[Optional[str]] = mapped_column(String(100))
    generation_reasoning: Mapped[Optional[str]] = mapped_column(Text)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float)
    iteration: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    
    # Performance metrics
    generation_time_seconds: Mapped[Optional[float]] = mapped_column(Float)
    token_count: Mapped[Optional[int]] = mapped_column(Integer)
    complexity_score: Mapped[Optional[float]] = mapped_column(Float)
    readability_score: Mapped[Optional[float]] = mapped_column(Float)
    
    # Vector embedding for similarity analysis
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(1536))
    
    # Optimization parameters
    parameters: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    ancestry: Mapped[List[str]] = mapped_column(JSON, default=list)  # Evolutionary history
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=datetime.utcnow,
        server_default=func.now()
    )
    
    # Relationships
    experiment: Mapped["OptimizationExperiment"] = relationship(
        "OptimizationExperiment", 
        back_populates="variants"
    )
    parent_prompt: Mapped["PromptTemplate"] = relationship(
        "PromptTemplate", 
        back_populates="variants"
    )
    evaluations: Mapped[List["PromptEvaluation"]] = relationship(
        "PromptEvaluation", 
        back_populates="prompt_variant",
        cascade="all, delete-orphan"
    )
    feedback: Mapped[List["PromptFeedback"]] = relationship(
        "PromptFeedback", 
        back_populates="prompt_variant",
        cascade="all, delete-orphan"
    )
    ab_tests_a: Mapped[List["ABTestResult"]] = relationship(
        "ABTestResult",
        foreign_keys="ABTestResult.prompt_a_id",
        back_populates="prompt_a"
    )
    ab_tests_b: Mapped[List["ABTestResult"]] = relationship(
        "ABTestResult",
        foreign_keys="ABTestResult.prompt_b_id",
        back_populates="prompt_b"
    )
    
    def __repr__(self) -> str:
        return f"<PromptVariant(id={self.id}, iteration={self.iteration}, confidence={self.confidence_score})>"


class PromptEvaluation(Base):
    """Performance evaluation of a prompt variant."""
    
    __tablename__ = "prompt_evaluations"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4,
        server_default=func.gen_random_uuid()
    )
    prompt_variant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("prompt_variants.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    test_case_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("prompt_test_cases.id", ondelete="SET NULL"),
        index=True
    )
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Evaluation details
    raw_output: Mapped[Optional[str]] = mapped_column(Text)
    expected_output: Mapped[Optional[str]] = mapped_column(Text)
    evaluation_context: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    evaluation_method: Mapped[Optional[str]] = mapped_column(String(100))
    evaluation_time_seconds: Mapped[Optional[float]] = mapped_column(Float)
    token_usage: Mapped[Dict[str, int]] = mapped_column(JSON, default=dict)
    cost_estimate: Mapped[Optional[float]] = mapped_column(Float)
    error_details: Mapped[Optional[str]] = mapped_column(Text)
    
    # Metadata
    evaluated_by: Mapped[Optional[str]] = mapped_column(String(255))
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=datetime.utcnow,
        server_default=func.now()
    )
    
    # Relationships
    prompt_variant: Mapped["PromptVariant"] = relationship(
        "PromptVariant", 
        back_populates="evaluations"
    )
    test_case: Mapped[Optional["PromptTestCase"]] = relationship(
        "PromptTestCase", 
        back_populates="evaluations"
    )
    
    def __repr__(self) -> str:
        return f"<PromptEvaluation(id={self.id}, metric='{self.metric_name}', value={self.metric_value})>"


class ABTestResult(Base):
    """A/B testing results for prompt variants."""
    
    __tablename__ = "ab_test_results"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4,
        server_default=func.gen_random_uuid()
    )
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("optimization_experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    test_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    prompt_a_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("prompt_variants.id", ondelete="CASCADE"),
        nullable=False
    )
    prompt_b_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("prompt_variants.id", ondelete="CASCADE"),
        nullable=False
    )
    
    # Test parameters
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False)
    significance_level: Mapped[float] = mapped_column(Float, default=0.05, server_default="0.05")
    
    # Statistical results
    p_value: Mapped[Optional[float]] = mapped_column(Float)
    effect_size: Mapped[Optional[float]] = mapped_column(Float)
    confidence_interval_lower: Mapped[Optional[float]] = mapped_column(Float)
    confidence_interval_upper: Mapped[Optional[float]] = mapped_column(Float)
    winner_variant_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("prompt_variants.id", ondelete="SET NULL")
    )
    test_power: Mapped[Optional[float]] = mapped_column(Float)
    
    # Detailed statistics
    mean_a: Mapped[Optional[float]] = mapped_column(Float)
    mean_b: Mapped[Optional[float]] = mapped_column(Float)
    std_a: Mapped[Optional[float]] = mapped_column(Float)
    std_b: Mapped[Optional[float]] = mapped_column(Float)
    test_statistic: Mapped[Optional[float]] = mapped_column(Float)
    degrees_of_freedom: Mapped[Optional[int]] = mapped_column(Integer)
    statistical_notes: Mapped[Optional[str]] = mapped_column(Text)
    
    # Timestamps
    test_completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=datetime.utcnow,
        server_default=func.now()
    )
    
    # Relationships
    experiment: Mapped["OptimizationExperiment"] = relationship(
        "OptimizationExperiment", 
        back_populates="ab_tests"
    )
    prompt_a: Mapped["PromptVariant"] = relationship(
        "PromptVariant",
        foreign_keys=[prompt_a_id],
        back_populates="ab_tests_a"
    )
    prompt_b: Mapped["PromptVariant"] = relationship(
        "PromptVariant",
        foreign_keys=[prompt_b_id],
        back_populates="ab_tests_b"
    )
    
    def __repr__(self) -> str:
        return f"<ABTestResult(id={self.id}, test='{self.test_name}', p_value={self.p_value})>"


class PromptFeedback(Base):
    """User feedback for prompt variants."""
    
    __tablename__ = "prompt_feedback"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4,
        server_default=func.gen_random_uuid()
    )
    prompt_variant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("prompt_variants.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    user_id: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    feedback_text: Mapped[Optional[str]] = mapped_column(Text)
    feedback_categories: Mapped[List[str]] = mapped_column(JSON, default=list)
    context_data: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    
    # Quality scores
    response_quality_score: Mapped[Optional[float]] = mapped_column(Float)
    relevance_score: Mapped[Optional[float]] = mapped_column(Float)
    clarity_score: Mapped[Optional[float]] = mapped_column(Float)
    usefulness_score: Mapped[Optional[float]] = mapped_column(Float)
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float)
    
    # Feedback weighting and validation
    feedback_weight: Mapped[float] = mapped_column(Float, default=1.0, server_default="1.0")
    is_validated: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    validation_notes: Mapped[Optional[str]] = mapped_column(Text)
    
    # Timestamps
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=datetime.utcnow,
        server_default=func.now()
    )
    
    # Relationships
    prompt_variant: Mapped["PromptVariant"] = relationship(
        "PromptVariant", 
        back_populates="feedback"
    )
    
    # Constraints
    __table_args__ = (
        CheckConstraint('rating >= 1 AND rating <= 5', name='check_rating_range'),
    )
    
    def __repr__(self) -> str:
        return f"<PromptFeedback(id={self.id}, rating={self.rating}, user_id='{self.user_id}')>"


class PromptTestCase(Base):
    """Test cases for systematic prompt evaluation."""
    
    __tablename__ = "prompt_test_cases"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4,
        server_default=func.gen_random_uuid()
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    domain: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    task_type: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    input_data: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    expected_output: Mapped[Optional[str]] = mapped_column(Text)
    evaluation_criteria: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    difficulty_level: Mapped[Optional[str]] = mapped_column(String(50))
    tags: Mapped[List[str]] = mapped_column(JSON, default=list)
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    created_by: Mapped[Optional[str]] = mapped_column(String(255))
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=datetime.utcnow,
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        server_default=func.now()
    )
    
    # Relationships
    evaluations: Mapped[List["PromptEvaluation"]] = relationship(
        "PromptEvaluation", 
        back_populates="test_case",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<PromptTestCase(id={self.id}, name='{self.name}', domain='{self.domain}')>"


class OptimizationMetric(Base):
    """System-wide optimization metrics for monitoring."""
    
    __tablename__ = "optimization_metrics"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4,
        server_default=func.gen_random_uuid()
    )
    experiment_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("optimization_experiments.id", ondelete="CASCADE"),
        index=True
    )
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    metric_type: Mapped[Optional[str]] = mapped_column(String(50))  # 'system', 'experiment', 'global'
    aggregation_period: Mapped[Optional[str]] = mapped_column(String(50))  # 'hour', 'day', 'week'
    tags: Mapped[Dict[str, str]] = mapped_column(JSON, default=dict)
    additional_data: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    
    # Timestamps
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=datetime.utcnow,
        server_default=func.now(),
        index=True
    )
    
    # Relationships
    experiment: Mapped[Optional["OptimizationExperiment"]] = relationship(
        "OptimizationExperiment", 
        back_populates="metrics"
    )
    
    def __repr__(self) -> str:
        return f"<OptimizationMetric(id={self.id}, metric='{self.metric_name}', value={self.metric_value})>"


# Additional indexes for performance optimization
Index('idx_prompt_templates_domain_task', PromptTemplate.domain, PromptTemplate.task_type)
Index('idx_prompt_templates_status_version', PromptTemplate.status, PromptTemplate.version)
Index('idx_optimization_experiments_status_method', OptimizationExperiment.status, OptimizationExperiment.optimization_method)
Index('idx_prompt_variants_experiment_iteration', PromptVariant.experiment_id, PromptVariant.iteration)
Index('idx_prompt_evaluations_metric_value', PromptEvaluation.metric_name, PromptEvaluation.metric_value)
Index('idx_ab_test_results_p_value', ABTestResult.p_value)
Index('idx_prompt_feedback_rating', PromptFeedback.rating)
Index('idx_prompt_test_cases_domain_task', PromptTestCase.domain, PromptTestCase.task_type)
Index('idx_optimization_metrics_name_type', OptimizationMetric.metric_name, OptimizationMetric.metric_type)