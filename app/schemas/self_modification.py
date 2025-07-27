"""
Self-Modification Engine Schemas

Pydantic schemas for API request/response validation and serialization
for the self-modification engine. Provides comprehensive type safety
and validation for all self-modification operations.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict, validator, field_validator
from enum import Enum

from app.models.self_modification import (
    ModificationSafety, ModificationStatus, ModificationType, SandboxExecutionType
)


# ============================================================================
# Enums (re-exported for schema consistency)
# ============================================================================

class ModificationSafetySchema(str, Enum):
    """Safety levels for code modifications."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class ModificationStatusSchema(str, Enum):
    """Status of modification sessions."""
    ANALYZING = "analyzing"
    SUGGESTIONS_READY = "suggestions_ready"
    APPLYING = "applying"
    APPLIED = "applied"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    ARCHIVED = "archived"


class ModificationTypeSchema(str, Enum):
    """Types of code modifications."""
    BUG_FIX = "bug_fix"
    PERFORMANCE = "performance"
    FEATURE_ADD = "feature_add"
    REFACTOR = "refactor"
    SECURITY_FIX = "security_fix"
    STYLE_IMPROVEMENT = "style_improvement"
    DEPENDENCY_UPDATE = "dependency_update"


class SandboxExecutionTypeSchema(str, Enum):
    """Types of sandbox executions."""
    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    SECURITY_SCAN = "security_scan"
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    LINTING = "linting"
    TYPE_CHECK = "type_check"


# ============================================================================
# Request Schemas
# ============================================================================

class AnalyzeCodebaseRequest(BaseModel):
    """Request to analyze codebase and generate modification suggestions."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    codebase_path: str = Field(
        ..., 
        min_length=1, 
        max_length=500,
        description="Path to the codebase to analyze"
    )
    modification_goals: List[str] = Field(
        ..., 
        min_length=1,
        description="List of modification goals (e.g., 'improve_performance', 'fix_bugs')"
    )
    safety_level: ModificationSafetySchema = Field(
        default=ModificationSafetySchema.CONSERVATIVE,
        description="Safety level for modifications"
    )
    repository_id: Optional[UUID] = Field(
        default=None,
        description="Optional GitHub repository ID if linked"
    )
    analysis_context: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional context for analysis"
    )
    include_patterns: Optional[List[str]] = Field(
        default=None,
        description="File patterns to include in analysis"
    )
    exclude_patterns: Optional[List[str]] = Field(
        default=None,
        description="File patterns to exclude from analysis"
    )
    
    @field_validator('modification_goals')
    @classmethod
    def validate_modification_goals(cls, v):
        valid_goals = {
            'improve_performance', 'fix_bugs', 'add_features', 'refactor_code',
            'improve_security', 'update_dependencies', 'improve_style'
        }
        for goal in v:
            if goal not in valid_goals:
                raise ValueError(f"Invalid modification goal: {goal}. Must be one of {valid_goals}")
        return v


class ApplyModificationsRequest(BaseModel):
    """Request to apply selected modifications."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    analysis_id: UUID = Field(..., description="ID of the analysis session")
    selected_modifications: List[UUID] = Field(
        ..., 
        min_length=1,
        description="List of modification IDs to apply"
    )
    approval_token: Optional[str] = Field(
        default=None,
        description="JWT token for human approval if required"
    )
    git_branch: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Git branch to apply modifications to"
    )
    commit_message: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Custom commit message for applied modifications"
    )
    dry_run: bool = Field(
        default=False,
        description="If true, validate modifications without applying them"
    )


class RollbackModificationRequest(BaseModel):
    """Request to rollback applied modifications."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    modification_id: UUID = Field(..., description="ID of the modification to rollback")
    rollback_reason: str = Field(
        ..., 
        min_length=1, 
        max_length=1000,
        description="Reason for rollback"
    )
    force_rollback: bool = Field(
        default=False,
        description="Force rollback even if there are conflicts"
    )


class ProvideFeedbackRequest(BaseModel):
    """Request to provide feedback on modifications."""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    modification_id: UUID = Field(..., description="ID of the modification")
    feedback_source: str = Field(
        ..., 
        max_length=100,
        description="Source of feedback (e.g., 'human', 'automated')"
    )
    feedback_type: str = Field(
        ..., 
        max_length=50,
        description="Type of feedback (e.g., 'rating', 'comment')"
    )
    rating: Optional[int] = Field(
        default=None, 
        ge=1, 
        le=5,
        description="Rating from 1-5"
    )
    feedback_text: Optional[str] = Field(
        default=None,
        max_length=5000,
        description="Detailed feedback text"
    )
    patterns_identified: Optional[List[str]] = Field(
        default=None,
        description="Code patterns identified in feedback"
    )
    improvement_suggestions: Optional[List[str]] = Field(
        default=None,
        description="Suggestions for future improvements"
    )


# ============================================================================
# Response Schemas
# ============================================================================

class ModificationSuggestion(BaseModel):
    """Individual modification suggestion."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    file_path: str
    modification_type: ModificationTypeSchema
    modification_reason: str
    llm_reasoning: Optional[str] = None
    safety_score: Decimal = Field(..., ge=0, le=1)
    complexity_score: Optional[Decimal] = Field(default=None, ge=0, le=1)
    performance_impact: Optional[Decimal] = None
    lines_added: Optional[int] = None
    lines_removed: Optional[int] = None
    functions_modified: Optional[int] = None
    dependencies_changed: List[str] = Field(default_factory=list)
    test_files_affected: List[str] = Field(default_factory=list)
    approval_required: bool = False
    original_content: Optional[str] = None
    modified_content: Optional[str] = None
    content_diff: Optional[str] = None


class AnalyzeCodebaseResponse(BaseModel):
    """Response from codebase analysis."""
    
    model_config = ConfigDict(from_attributes=True)
    
    analysis_id: UUID
    status: ModificationStatusSchema
    total_suggestions: int
    suggestions: List[ModificationSuggestion]
    codebase_summary: Dict[str, Any] = Field(default_factory=dict)
    analysis_metadata: Dict[str, Any] = Field(default_factory=dict)
    estimated_completion_time: Optional[int] = None  # minutes
    created_at: datetime


class ApplyModificationsResponse(BaseModel):
    """Response from applying modifications."""
    
    model_config = ConfigDict(from_attributes=True)
    
    session_id: UUID
    status: str
    applied_modifications: List[UUID]
    failed_modifications: List[UUID] = Field(default_factory=list)
    git_commit_hash: Optional[str] = None
    git_branch: Optional[str] = None
    rollback_commit_hash: Optional[str] = None
    applied_at: Optional[datetime] = None
    error_messages: List[str] = Field(default_factory=list)
    performance_impact: Optional[Decimal] = None


class RollbackModificationResponse(BaseModel):
    """Response from rolling back modifications."""
    
    model_config = ConfigDict(from_attributes=True)
    
    success: bool
    modification_id: UUID
    restored_commit_hash: Optional[str] = None
    rollback_reason: str
    rollback_at: datetime
    error_message: Optional[str] = None


class ModificationMetricResponse(BaseModel):
    """Individual metric measurement."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    metric_name: str
    metric_category: str
    baseline_value: Optional[Decimal] = None
    modified_value: Optional[Decimal] = None
    improvement_percentage: Optional[Decimal] = None
    measurement_unit: Optional[str] = None
    measurement_context: Optional[str] = None
    confidence_score: Optional[Decimal] = None
    statistical_significance: Optional[bool] = None
    measured_at: datetime


class SandboxExecutionResponse(BaseModel):
    """Sandbox execution result."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    execution_type: SandboxExecutionTypeSchema
    command: str
    exit_code: Optional[int] = None
    execution_time_ms: Optional[int] = None
    memory_usage_mb: Optional[int] = None
    cpu_usage_percent: Optional[Decimal] = None
    network_attempts: int = 0
    security_violations: List[Dict[str, Any]] = Field(default_factory=list)
    test_results: Dict[str, Any] = Field(default_factory=dict)
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    executed_at: datetime


class ModificationSessionResponse(BaseModel):
    """Complete modification session details."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    agent_id: UUID
    repository_id: Optional[UUID] = None
    codebase_path: str
    modification_goals: List[str]
    safety_level: ModificationSafetySchema
    status: ModificationStatusSchema
    total_suggestions: int
    applied_modifications: int
    success_rate: Optional[Decimal] = None
    performance_improvement: Optional[Decimal] = None
    error_message: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    # Nested relationships
    modifications: List[ModificationSuggestion] = Field(default_factory=list)
    sandbox_executions: List[SandboxExecutionResponse] = Field(default_factory=list)


class ModificationSessionSummary(BaseModel):
    """Summary of modification session (for list endpoints)."""
    
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    agent_id: UUID
    codebase_path: str
    status: ModificationStatusSchema
    safety_level: ModificationSafetySchema
    total_suggestions: int
    applied_modifications: int
    success_rate: Optional[Decimal] = None
    performance_improvement: Optional[Decimal] = None
    started_at: datetime
    completed_at: Optional[datetime] = None


class GetSessionsResponse(BaseModel):
    """Response for listing modification sessions."""
    
    sessions: List[ModificationSessionSummary]
    total: int
    page: int = 1
    page_size: int = 20
    has_next: bool = False


class ModificationMetricsResponse(BaseModel):
    """Aggregated metrics for modifications."""
    
    model_config = ConfigDict(from_attributes=True)
    
    session_id: Optional[UUID] = None
    modification_id: Optional[UUID] = None
    metrics: List[ModificationMetricResponse]
    overall_performance_improvement: Optional[Decimal] = None
    overall_success_rate: Optional[Decimal] = None
    total_modifications: int
    successful_modifications: int
    failed_modifications: int


class SystemHealthResponse(BaseModel):
    """System health status for self-modification engine."""
    
    sandbox_environment_healthy: bool
    git_integration_healthy: bool
    modification_queue_size: int
    active_sessions: int
    average_success_rate: Optional[Decimal] = None
    average_performance_improvement: Optional[Decimal] = None
    last_successful_modification: Optional[datetime] = None
    system_uptime_hours: Optional[float] = None


# ============================================================================
# Error Response Schemas
# ============================================================================

class ErrorDetail(BaseModel):
    """Individual error detail."""
    
    code: str
    message: str
    field: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Standard error response format."""
    
    error: bool = True
    error_type: str
    message: str
    details: List[ErrorDetail] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


# ============================================================================
# Utility Schemas
# ============================================================================

class PaginationParams(BaseModel):
    """Standard pagination parameters."""
    
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")
    sort_by: Optional[str] = Field(default=None, description="Field to sort by")
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$", description="Sort order")


class FilterParams(BaseModel):
    """Filtering parameters for sessions and modifications."""
    
    status: Optional[ModificationStatusSchema] = None
    safety_level: Optional[ModificationSafetySchema] = None
    modification_type: Optional[ModificationTypeSchema] = None
    agent_id: Optional[UUID] = None
    repository_id: Optional[UUID] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_success_rate: Optional[float] = Field(default=None, ge=0, le=100)
    min_performance_improvement: Optional[float] = Field(default=None)


# Export all schemas
__all__ = [
    # Enums
    "ModificationSafetySchema",
    "ModificationStatusSchema", 
    "ModificationTypeSchema",
    "SandboxExecutionTypeSchema",
    
    # Request schemas
    "AnalyzeCodebaseRequest",
    "ApplyModificationsRequest",
    "RollbackModificationRequest",
    "ProvideFeedbackRequest",
    
    # Response schemas
    "ModificationSuggestion",
    "AnalyzeCodebaseResponse",
    "ApplyModificationsResponse",
    "RollbackModificationResponse",
    "ModificationMetricResponse",
    "SandboxExecutionResponse",
    "ModificationSessionResponse",
    "ModificationSessionSummary",
    "GetSessionsResponse",
    "ModificationMetricsResponse",
    "SystemHealthResponse",
    
    # Error schemas
    "ErrorDetail",
    "ErrorResponse",
    
    # Utility schemas
    "PaginationParams",
    "FilterParams",
]