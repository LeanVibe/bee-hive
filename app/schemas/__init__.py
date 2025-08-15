"""Pydantic schemas for API request/response models."""

from .agent import (
    AgentCreate, AgentUpdate, AgentResponse, AgentListResponse,
    AgentCapabilityCreate, AgentStatsResponse
)
from .session import (
    SessionCreate, SessionUpdate, SessionResponse, SessionListResponse
)
from .task import (
    TaskCreate, TaskUpdate, TaskResponse, TaskListResponse
)
from .workflow import (
    WorkflowCreate, WorkflowUpdate, WorkflowResponse, WorkflowListResponse,
    WorkflowTaskAssignment, WorkflowExecutionRequest, WorkflowProgressResponse,
    WorkflowValidationResponse, WorkflowStatsResponse
)
from .context import (
    ContextCreate, ContextUpdate, ContextResponse, ContextListResponse
)
from .self_modification import (
    AnalyzeCodebaseRequest, ApplyModificationsRequest, RollbackModificationRequest,
    ProvideFeedbackRequest, ModificationSuggestion, AnalyzeCodebaseResponse,
    ApplyModificationsResponse, RollbackModificationResponse, ModificationMetricResponse,
    SandboxExecutionResponse, ModificationSessionResponse, ModificationSessionSummary,
    GetSessionsResponse, ModificationMetricsResponse, SystemHealthResponse,
    ErrorDetail, ErrorResponse, PaginationParams, FilterParams,
    ModificationSafetySchema, ModificationStatusSchema, ModificationTypeSchema,
    SandboxExecutionTypeSchema
)
from .prompt_optimization import (
    PromptTemplateCreate, PromptTemplateUpdate, PromptTemplateResponse, PromptTemplateListResponse,
    OptimizationExperimentCreate, OptimizationExperimentUpdate, OptimizationExperimentResponse,
    OptimizationExperimentListResponse, PromptVariantResponse, PromptVariantListResponse,
    PromptEvaluationCreate, PromptEvaluationResponse, PromptEvaluationBatchResponse,
    ABTestCreate, ABTestResponse, PromptFeedbackCreate, PromptFeedbackResponse,
    PromptTestCaseCreate, PromptTestCaseResponse, PromptGenerationRequest, PromptGenerationResponse,
    OptimizationMetricResponse, SystemMetricsResponse, PromptOptimizationError,
    BulkEvaluationRequest, BulkEvaluationResponse, PromptSearchRequest
)
from .project_index import (
    ProjectIndexCreate, ProjectIndexUpdate, ProjectIndexResponse, ProjectIndexListResponse,
    FileEntryCreate, FileEntryUpdate, FileEntryResponse, FileEntryListResponse,
    DependencyRelationshipCreate, DependencyRelationshipUpdate, DependencyRelationshipResponse, DependencyRelationshipListResponse,
    IndexSnapshotCreate, IndexSnapshotResponse, IndexSnapshotListResponse,
    AnalysisSessionCreate, AnalysisSessionUpdate, AnalysisSessionResponse, AnalysisSessionListResponse,
    ProjectIndexFilter, FileEntryFilter, DependencyRelationshipFilter, AnalysisSessionFilter,
    BulkFileCreate, BulkDependencyCreate, BulkOperationResponse,
    ProjectStatistics, DependencyGraph, DependencyGraphNode, DependencyGraphEdge, AnalysisProgress
)

__all__ = [
    "AgentCreate", "AgentUpdate", "AgentResponse", "AgentListResponse",
    "AgentCapabilityCreate", "AgentStatsResponse",
    "SessionCreate", "SessionUpdate", "SessionResponse", "SessionListResponse",
    "TaskCreate", "TaskUpdate", "TaskResponse", "TaskListResponse", 
    "WorkflowCreate", "WorkflowUpdate", "WorkflowResponse", "WorkflowListResponse",
    "WorkflowTaskAssignment", "WorkflowExecutionRequest", "WorkflowProgressResponse",
    "WorkflowValidationResponse", "WorkflowStatsResponse",
    "ContextCreate", "ContextUpdate", "ContextResponse", "ContextListResponse",
    "AnalyzeCodebaseRequest", "ApplyModificationsRequest", "RollbackModificationRequest",
    "ProvideFeedbackRequest", "ModificationSuggestion", "AnalyzeCodebaseResponse",
    "ApplyModificationsResponse", "RollbackModificationResponse", "ModificationMetricResponse",
    "SandboxExecutionResponse", "ModificationSessionResponse", "ModificationSessionSummary",
    "GetSessionsResponse", "ModificationMetricsResponse", "SystemHealthResponse",
    "ErrorDetail", "ErrorResponse", "PaginationParams", "FilterParams",
    "ModificationSafetySchema", "ModificationStatusSchema", "ModificationTypeSchema",
    "SandboxExecutionTypeSchema",
    "PromptTemplateCreate", "PromptTemplateUpdate", "PromptTemplateResponse", "PromptTemplateListResponse",
    "OptimizationExperimentCreate", "OptimizationExperimentUpdate", "OptimizationExperimentResponse",
    "OptimizationExperimentListResponse", "PromptVariantResponse", "PromptVariantListResponse",
    "PromptEvaluationCreate", "PromptEvaluationResponse", "PromptEvaluationBatchResponse",
    "ABTestCreate", "ABTestResponse", "PromptFeedbackCreate", "PromptFeedbackResponse",
    "PromptTestCaseCreate", "PromptTestCaseResponse", "PromptGenerationRequest", "PromptGenerationResponse",
    "OptimizationMetricResponse", "SystemMetricsResponse", "PromptOptimizationError",
    "BulkEvaluationRequest", "BulkEvaluationResponse", "PromptSearchRequest",
    "ProjectIndexCreate", "ProjectIndexUpdate", "ProjectIndexResponse", "ProjectIndexListResponse",
    "FileEntryCreate", "FileEntryUpdate", "FileEntryResponse", "FileEntryListResponse",
    "DependencyRelationshipCreate", "DependencyRelationshipUpdate", "DependencyRelationshipResponse", "DependencyRelationshipListResponse",
    "IndexSnapshotCreate", "IndexSnapshotResponse", "IndexSnapshotListResponse",
    "AnalysisSessionCreate", "AnalysisSessionUpdate", "AnalysisSessionResponse", "AnalysisSessionListResponse",
    "ProjectIndexFilter", "FileEntryFilter", "DependencyRelationshipFilter", "AnalysisSessionFilter",
    "BulkFileCreate", "BulkDependencyCreate", "BulkOperationResponse",
    "ProjectStatistics", "DependencyGraph", "DependencyGraphNode", "DependencyGraphEdge", "AnalysisProgress"
]