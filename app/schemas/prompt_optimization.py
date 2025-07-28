"""
Pydantic schemas for Prompt Optimization System API.

Defines request/response models for prompt optimization endpoints,
validation rules, and data serialization formats.
"""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, validator, root_validator

from ..models.prompt_optimization import (
    PromptStatus, ExperimentStatus, OptimizationMethod
)


# Base schemas
class BasePromptOptimizationSchema(BaseModel):
    """Base schema with common configuration."""
    
    class Config:
        from_attributes = True
        use_enum_values = True
        validate_assignment = True


# Prompt Template schemas
class PromptTemplateCreate(BasePromptOptimizationSchema):
    """Schema for creating a new prompt template."""
    
    name: str = Field(..., min_length=1, max_length=255, description="Template name")
    task_type: Optional[str] = Field(None, max_length=100, description="Type of task this prompt performs")
    domain: Optional[str] = Field(None, max_length=100, description="Domain or subject area")
    template_content: str = Field(..., min_length=1, description="The prompt template content")
    template_variables: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Variables used in template")
    description: Optional[str] = Field(None, description="Template description")
    tags: Optional[List[str]] = Field(default_factory=list, description="Tags for categorization")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    created_by: Optional[str] = Field(None, max_length=255, description="Creator identifier")


class PromptTemplateUpdate(BasePromptOptimizationSchema):
    """Schema for updating a prompt template."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    task_type: Optional[str] = Field(None, max_length=100)
    domain: Optional[str] = Field(None, max_length=100)
    template_content: Optional[str] = Field(None, min_length=1)
    template_variables: Optional[Dict[str, Any]] = None
    status: Optional[PromptStatus] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class PromptTemplateResponse(BasePromptOptimizationSchema):
    """Schema for prompt template responses."""
    
    id: uuid.UUID
    name: str
    task_type: Optional[str]
    domain: Optional[str]
    template_content: str
    template_variables: Dict[str, Any]
    version: int
    status: PromptStatus
    created_by: Optional[str]
    description: Optional[str]
    tags: List[str]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    last_used_at: Optional[datetime]


class PromptTemplateListResponse(BasePromptOptimizationSchema):
    """Schema for paginated prompt template list."""
    
    templates: List[PromptTemplateResponse]
    total: int
    offset: int
    limit: int


# Optimization Experiment schemas
class OptimizationExperimentCreate(BasePromptOptimizationSchema):
    """Schema for creating an optimization experiment."""
    
    experiment_name: str = Field(..., min_length=1, max_length=255, description="Experiment name")
    base_prompt_id: uuid.UUID = Field(..., description="Base prompt template to optimize")
    optimization_method: OptimizationMethod = Field(..., description="Optimization method to use")
    target_metrics: Dict[str, float] = Field(..., description="Target performance metrics")
    experiment_config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Experiment configuration")
    max_iterations: Optional[int] = Field(50, ge=1, le=1000, description="Maximum iterations")
    convergence_threshold: Optional[float] = Field(0.01, gt=0, le=1, description="Convergence threshold")
    early_stopping: Optional[bool] = Field(True, description="Enable early stopping")
    description: Optional[str] = Field(None, description="Experiment description")
    created_by: Optional[str] = Field(None, max_length=255, description="Creator identifier")
    
    @validator('target_metrics')
    def validate_target_metrics(cls, v):
        if not v:
            raise ValueError("At least one target metric must be specified")
        for metric_name, target_value in v.items():
            if not isinstance(target_value, (int, float)):
                raise ValueError(f"Target value for {metric_name} must be numeric")
            if target_value < 0 or target_value > 1:
                raise ValueError(f"Target value for {metric_name} must be between 0 and 1")
        return v


class OptimizationExperimentUpdate(BasePromptOptimizationSchema):
    """Schema for updating an experiment."""
    
    experiment_name: Optional[str] = Field(None, min_length=1, max_length=255)
    status: Optional[ExperimentStatus] = None
    target_metrics: Optional[Dict[str, float]] = None
    experiment_config: Optional[Dict[str, Any]] = None
    max_iterations: Optional[int] = Field(None, ge=1, le=1000)
    convergence_threshold: Optional[float] = Field(None, gt=0, le=1)
    early_stopping: Optional[bool] = None
    description: Optional[str] = None


class OptimizationExperimentResponse(BasePromptOptimizationSchema):
    """Schema for optimization experiment responses."""
    
    id: uuid.UUID
    experiment_name: str
    base_prompt_id: uuid.UUID
    optimization_method: OptimizationMethod
    target_metrics: Dict[str, float]
    experiment_config: Dict[str, Any]
    status: ExperimentStatus
    progress_percentage: float
    current_iteration: int
    max_iterations: int
    best_score: Optional[float]
    baseline_score: Optional[float]
    improvement_percentage: Optional[float]
    convergence_threshold: float
    early_stopping: bool
    created_by: Optional[str]
    description: Optional[str]
    error_message: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime


class OptimizationExperimentListResponse(BasePromptOptimizationSchema):
    """Schema for paginated experiment list."""
    
    experiments: List[OptimizationExperimentResponse]
    total: int
    offset: int
    limit: int


# Prompt Variant schemas
class PromptVariantResponse(BasePromptOptimizationSchema):
    """Schema for prompt variant responses."""
    
    id: uuid.UUID
    experiment_id: uuid.UUID
    parent_prompt_id: uuid.UUID
    variant_content: str
    generation_method: Optional[str]
    generation_reasoning: Optional[str]
    confidence_score: Optional[float]
    iteration: int
    generation_time_seconds: Optional[float]
    token_count: Optional[int]
    complexity_score: Optional[float]
    readability_score: Optional[float]
    parameters: Dict[str, Any]
    ancestry: List[str]
    created_at: datetime


class PromptVariantListResponse(BasePromptOptimizationSchema):
    """Schema for paginated variant list."""
    
    variants: List[PromptVariantResponse]
    total: int
    offset: int
    limit: int


# Prompt Evaluation schemas
class PromptEvaluationCreate(BasePromptOptimizationSchema):
    """Schema for creating a prompt evaluation."""
    
    prompt_variant_id: uuid.UUID = Field(..., description="Variant to evaluate")
    test_case_id: Optional[uuid.UUID] = Field(None, description="Test case to use")
    evaluation_metrics: List[str] = Field(..., min_items=1, description="Metrics to evaluate")
    evaluation_context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Evaluation context")


class PromptEvaluationResponse(BasePromptOptimizationSchema):
    """Schema for prompt evaluation responses."""
    
    id: uuid.UUID
    prompt_variant_id: uuid.UUID
    test_case_id: Optional[uuid.UUID]
    metric_name: str
    metric_value: float
    raw_output: Optional[str]
    expected_output: Optional[str]
    evaluation_context: Dict[str, Any]
    evaluation_method: Optional[str]
    evaluation_time_seconds: Optional[float]
    token_usage: Dict[str, int]
    cost_estimate: Optional[float]
    error_details: Optional[str]
    evaluated_by: Optional[str]
    evaluated_at: datetime


class PromptEvaluationBatchResponse(BasePromptOptimizationSchema):
    """Schema for batch evaluation response."""
    
    evaluations: List[PromptEvaluationResponse]
    performance_score: float = Field(..., description="Overall performance score")
    detailed_metrics: Dict[str, float] = Field(..., description="Detailed metric scores")


# A/B Testing schemas
class ABTestCreate(BasePromptOptimizationSchema):
    """Schema for creating an A/B test."""
    
    experiment_id: uuid.UUID = Field(..., description="Parent experiment")
    test_name: str = Field(..., min_length=1, max_length=255, description="Test name")
    prompt_a_id: uuid.UUID = Field(..., description="First prompt variant")
    prompt_b_id: uuid.UUID = Field(..., description="Second prompt variant")
    sample_size: int = Field(..., ge=10, le=10000, description="Sample size for test")
    significance_level: Optional[float] = Field(0.05, gt=0, lt=1, description="Statistical significance level")
    
    @validator('prompt_a_id', 'prompt_b_id')
    def validate_different_prompts(cls, v, values):
        if 'prompt_a_id' in values and v == values['prompt_a_id']:
            raise ValueError("Prompt A and B must be different")
        return v


class ABTestResponse(BasePromptOptimizationSchema):
    """Schema for A/B test responses."""
    
    id: uuid.UUID
    experiment_id: uuid.UUID
    test_name: str
    prompt_a_id: uuid.UUID
    prompt_b_id: uuid.UUID
    sample_size: int
    significance_level: float
    p_value: Optional[float]
    effect_size: Optional[float]
    confidence_interval_lower: Optional[float]
    confidence_interval_upper: Optional[float]
    winner_variant_id: Optional[uuid.UUID]
    test_power: Optional[float]
    mean_a: Optional[float]
    mean_b: Optional[float]
    std_a: Optional[float]
    std_b: Optional[float]
    test_statistic: Optional[float]
    degrees_of_freedom: Optional[int]
    statistical_notes: Optional[str]
    test_completed_at: Optional[datetime]
    created_at: datetime


# Feedback schemas
class PromptFeedbackCreate(BasePromptOptimizationSchema):
    """Schema for submitting prompt feedback."""
    
    prompt_variant_id: uuid.UUID = Field(..., description="Variant being rated")
    user_id: Optional[str] = Field(None, max_length=255, description="User identifier")
    session_id: Optional[str] = Field(None, max_length=255, description="Session identifier")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    feedback_text: Optional[str] = Field(None, description="Textual feedback")
    feedback_categories: Optional[List[str]] = Field(default_factory=list, description="Feedback categories")
    context_data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Context information")


class PromptFeedbackResponse(BasePromptOptimizationSchema):
    """Schema for feedback responses."""
    
    id: uuid.UUID
    prompt_variant_id: uuid.UUID
    user_id: Optional[str]
    session_id: Optional[str]
    rating: int
    feedback_text: Optional[str]
    feedback_categories: List[str]
    context_data: Dict[str, Any]
    response_quality_score: Optional[float]
    relevance_score: Optional[float]
    clarity_score: Optional[float]
    usefulness_score: Optional[float]
    sentiment_score: Optional[float]
    feedback_weight: float
    is_validated: bool
    validation_notes: Optional[str]
    submitted_at: datetime
    status: str = Field(..., description="Processing status")
    influence_weight: float = Field(..., description="Weight of this feedback in optimization")


# Test Case schemas
class PromptTestCaseCreate(BasePromptOptimizationSchema):
    """Schema for creating a test case."""
    
    name: str = Field(..., min_length=1, max_length=255, description="Test case name")
    description: Optional[str] = Field(None, description="Test case description")
    domain: Optional[str] = Field(None, max_length=100, description="Domain")
    task_type: Optional[str] = Field(None, max_length=100, description="Task type")
    input_data: Dict[str, Any] = Field(..., description="Input data for test")
    expected_output: Optional[str] = Field(None, description="Expected output")
    evaluation_criteria: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Evaluation criteria")
    difficulty_level: Optional[str] = Field(None, max_length=50, description="Difficulty level")
    tags: Optional[List[str]] = Field(default_factory=list, description="Tags")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    created_by: Optional[str] = Field(None, max_length=255, description="Creator identifier")


class PromptTestCaseResponse(BasePromptOptimizationSchema):
    """Schema for test case responses."""
    
    id: uuid.UUID
    name: str
    description: Optional[str]
    domain: Optional[str]
    task_type: Optional[str]
    input_data: Dict[str, Any]
    expected_output: Optional[str]
    evaluation_criteria: Dict[str, Any]
    difficulty_level: Optional[str]
    tags: List[str]
    metadata: Dict[str, Any]
    is_active: bool
    created_by: Optional[str]
    created_at: datetime
    updated_at: datetime


# Prompt generation request schemas
class PromptGenerationRequest(BasePromptOptimizationSchema):
    """Schema for prompt generation requests."""
    
    task_description: str = Field(..., min_length=1, description="Description of the task")
    domain: Optional[str] = Field(None, max_length=100, description="Domain or subject area")
    performance_goals: List[str] = Field(..., min_items=1, description="Performance goals")
    baseline_examples: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Example inputs/outputs")
    constraints: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Generation constraints")
    optimization_preferences: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optimization preferences")
    
    @validator('performance_goals')
    def validate_performance_goals(cls, v):
        valid_goals = ['accuracy', 'token_efficiency', 'user_satisfaction', 'clarity', 'relevance', 'coherence']
        for goal in v:
            if goal not in valid_goals:
                raise ValueError(f"Invalid performance goal: {goal}. Valid goals: {valid_goals}")
        return v


class PromptGenerationResponse(BasePromptOptimizationSchema):
    """Schema for prompt generation responses."""
    
    prompt_candidates: List[Dict[str, Any]] = Field(..., description="Generated prompt candidates")
    experiment_id: uuid.UUID = Field(..., description="Created experiment ID")
    generation_metadata: Dict[str, Any] = Field(..., description="Generation process metadata")


# System metrics and monitoring schemas
class OptimizationMetricResponse(BasePromptOptimizationSchema):
    """Schema for optimization metrics."""
    
    id: uuid.UUID
    experiment_id: Optional[uuid.UUID]
    metric_name: str
    metric_value: float
    metric_type: Optional[str]
    aggregation_period: Optional[str]
    tags: Dict[str, str]
    additional_data: Dict[str, Any]
    recorded_at: datetime


class SystemMetricsResponse(BasePromptOptimizationSchema):
    """Schema for system-wide metrics."""
    
    active_experiments: int
    total_prompts_optimized: int
    average_improvement_percentage: float
    successful_optimizations: int
    failed_optimizations: int
    total_evaluations_performed: int
    average_optimization_time_hours: float
    metrics_by_method: Dict[str, Dict[str, float]]


# Error response schemas
class PromptOptimizationError(BasePromptOptimizationSchema):
    """Schema for error responses."""
    
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


# Bulk operation schemas
class BulkEvaluationRequest(BasePromptOptimizationSchema):
    """Schema for bulk evaluation requests."""
    
    prompt_variant_ids: List[uuid.UUID] = Field(..., min_items=1, max_items=100, description="Variants to evaluate")
    test_case_ids: Optional[List[uuid.UUID]] = Field(None, description="Test cases to use")
    evaluation_metrics: List[str] = Field(..., min_items=1, description="Metrics to evaluate")
    evaluation_config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Evaluation configuration")


class BulkEvaluationResponse(BasePromptOptimizationSchema):
    """Schema for bulk evaluation responses."""
    
    total_evaluations: int
    successful_evaluations: int
    failed_evaluations: int
    evaluation_results: List[PromptEvaluationResponse]
    error_details: List[Dict[str, Any]]


# Search and filter schemas
class PromptSearchRequest(BasePromptOptimizationSchema):
    """Schema for searching prompts."""
    
    query: Optional[str] = Field(None, description="Search query")
    domain: Optional[str] = Field(None, description="Domain filter")
    task_type: Optional[str] = Field(None, description="Task type filter")
    status: Optional[PromptStatus] = Field(None, description="Status filter")
    tags: Optional[List[str]] = Field(None, description="Tag filters")
    min_performance_score: Optional[float] = Field(None, ge=0, le=1, description="Minimum performance score")
    created_after: Optional[datetime] = Field(None, description="Created after date")
    created_before: Optional[datetime] = Field(None, description="Created before date")
    limit: Optional[int] = Field(50, ge=1, le=100, description="Result limit")
    offset: Optional[int] = Field(0, ge=0, description="Result offset")