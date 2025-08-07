"""
GitHub Integration Pydantic Schemas

Comprehensive schema definitions for advanced GitHub integration API endpoints
including AI code review, automated testing, conflict resolution, and workflow automation.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator


# Enum definitions
class ReviewType(str, Enum):
    """Code review types."""
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    MAINTAINABILITY = "maintainability"
    COMPREHENSIVE = "comprehensive"


class TestSuiteType(str, Enum):
    """Test suite types."""
    UNIT = "unit"
    INTEGRATION = "integration"
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    SECURITY = "security"
    E2E = "e2e"
    SMOKE = "smoke"
    REGRESSION = "regression"


class ConflictResolutionStrategy(str, Enum):
    """Merge conflict resolution strategies."""
    AUTOMATIC = "automatic"
    PREFER_OURS = "prefer_ours"
    PREFER_THEIRS = "prefer_theirs"
    INTELLIGENT_MERGE = "intelligent_merge"
    MANUAL_REQUIRED = "manual_required"
    SEMANTIC_MERGE = "semantic_merge"


class WorkflowTriggerType(str, Enum):
    """Workflow trigger types."""
    PR_CREATED = "pr_created"
    PR_UPDATED = "pr_updated"
    COMMIT_PUSHED = "commit_pushed"
    REVIEW_REQUESTED = "review_requested"
    MANUAL_TRIGGER = "manual_trigger"
    SCHEDULED = "scheduled"
    WEBHOOK_EVENT = "webhook_event"


class WorkflowStageType(str, Enum):
    """Workflow execution stages."""
    INITIATED = "initiated"
    CODE_ANALYSIS = "code_analysis"
    AUTOMATED_TESTING = "automated_testing"
    SECURITY_SCANNING = "security_scanning"
    CODE_FORMATTING = "code_formatting"
    DOCUMENTATION_GENERATION = "documentation_generation"
    QUALITY_GATES = "quality_gates"
    PEER_REVIEW = "peer_review"
    APPROVAL = "approval"
    MERGE_PREPARATION = "merge_preparation"
    AUTOMATED_MERGE = "automated_merge"
    POST_MERGE_ACTIONS = "post_merge_actions"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class HealthStatus(str, Enum):
    """Repository health status."""
    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


# Base schemas
class BaseResponse(BaseModel):
    """Base response model with common fields."""
    success: bool = Field(..., description="Whether the operation was successful")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    class Config:
        use_enum_values = True


class ErrorDetail(BaseModel):
    """Error detail information."""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class Finding(BaseModel):
    """Code review finding."""
    category: str = Field(..., description="Finding category (security, performance, style)")
    severity: str = Field(..., description="Finding severity (critical, high, medium, low, info)")
    type: str = Field(..., description="Specific finding type")
    file: str = Field(..., description="File path")
    line: int = Field(..., description="Line number")
    column: Optional[int] = Field(None, description="Column number")
    message: str = Field(..., description="Finding description")
    suggestion: Optional[str] = Field(None, description="Improvement suggestion")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")


class Recommendation(BaseModel):
    """Improvement recommendation."""
    type: str = Field(..., description="Recommendation type")
    priority: str = Field(..., description="Priority level (critical, high, medium, low)")
    title: str = Field(..., description="Recommendation title")
    description: str = Field(..., description="Detailed description")
    recommendation: str = Field(..., description="Recommended action")
    impact: Optional[str] = Field(None, description="Expected impact")
    effort_estimate: Optional[str] = Field(None, description="Effort estimate")


# AI Code Review Schemas
class CodeReviewRequest(BaseModel):
    """AI code review analysis request."""
    pull_request_id: UUID = Field(..., description="Pull request ID to analyze")
    review_types: List[ReviewType] = Field(
        default=[ReviewType.SECURITY, ReviewType.PERFORMANCE, ReviewType.STYLE],
        description="Types of code review analysis to perform"
    )
    include_suggestions: bool = Field(True, description="Include improvement suggestions")
    post_to_github: bool = Field(True, description="Post review results to GitHub PR")
    
    @validator('review_types')
    def validate_review_types(cls, v):
        if not v:
            raise ValueError("At least one review type must be specified")
        return v


class CodeReviewResponse(BaseResponse):
    """AI code review analysis response."""
    review_id: str = Field(..., description="Unique review ID")
    pull_request_id: UUID = Field(..., description="Pull request ID")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall code quality score (0.0-1.0)")
    approved: bool = Field(..., description="Whether the code review is approved")
    changes_requested: bool = Field(..., description="Whether changes are requested")
    findings_count: int = Field(..., ge=0, description="Total number of findings")
    categorized_findings: Dict[str, int] = Field(..., description="Findings categorized by type")
    suggestions_count: int = Field(..., ge=0, description="Number of improvement suggestions")
    github_posted: bool = Field(..., description="Whether results were posted to GitHub")
    review_duration_seconds: int = Field(..., ge=0, description="Review analysis duration")
    files_analyzed: int = Field(..., ge=0, description="Number of files analyzed")
    analysis_summary: str = Field(..., description="Human-readable analysis summary")
    recommendations: List[Recommendation] = Field(default=[], description="Prioritized recommendations")
    findings: Optional[List[Finding]] = Field(None, description="Detailed findings list")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall confidence in analysis")


# Automated Testing Schemas
class TestExecutionRequest(BaseModel):
    """Automated test execution request."""
    pull_request_id: UUID = Field(..., description="Pull request ID to test")
    test_suites: List[TestSuiteType] = Field(
        default=[TestSuiteType.UNIT, TestSuiteType.INTEGRATION, TestSuiteType.SECURITY],
        description="Test suites to execute"
    )
    force_run: bool = Field(False, description="Force test execution even if already running")
    wait_for_completion: bool = Field(False, description="Wait for test completion before responding")
    timeout_minutes: Optional[int] = Field(60, ge=1, le=300, description="Test execution timeout in minutes")
    
    @validator('test_suites')
    def validate_test_suites(cls, v):
        if not v:
            raise ValueError("At least one test suite must be specified")
        return v


class TestResult(BaseModel):
    """Individual test result."""
    name: str = Field(..., description="Test name")
    status: str = Field(..., description="Test status (passed, failed, skipped)")
    duration: float = Field(..., ge=0, description="Test execution time in seconds")
    message: Optional[str] = Field(None, description="Test result message")
    failure_type: Optional[str] = Field(None, description="Failure type if test failed")


class TestSuiteResult(BaseModel):
    """Test suite execution result."""
    suite_name: str = Field(..., description="Test suite name")
    suite_type: TestSuiteType = Field(..., description="Test suite type")
    status: str = Field(..., description="Suite status (passed, failed, skipped)")
    total_tests: int = Field(..., ge=0, description="Total number of tests")
    passed_tests: int = Field(..., ge=0, description="Number of passed tests")
    failed_tests: int = Field(..., ge=0, description="Number of failed tests")
    skipped_tests: int = Field(..., ge=0, description="Number of skipped tests")
    duration: float = Field(..., ge=0, description="Suite execution time in seconds")
    coverage: Optional[float] = Field(None, ge=0.0, le=100.0, description="Code coverage percentage")
    test_results: Optional[List[TestResult]] = Field(None, description="Individual test results")


class FailureAnalysis(BaseModel):
    """Test failure analysis."""
    failure_categories: Dict[str, List[Dict[str, Any]]] = Field(..., description="Categorized failures")
    common_patterns: List[Dict[str, Any]] = Field(..., description="Common failure patterns")
    affected_areas: List[str] = Field(..., description="Code areas affected by failures")
    recommendations: List[Recommendation] = Field(..., description="Failure resolution recommendations")
    retry_suggestions: List[Dict[str, Any]] = Field(..., description="Retry recommendations")
    failure_severity: str = Field(..., description="Overall failure severity")


class TestExecutionResponse(BaseResponse):
    """Automated test execution response."""
    test_run_id: str = Field(..., description="Unique test run ID")
    pull_request_id: UUID = Field(..., description="Pull request ID")
    status: str = Field(..., description="Test execution status")
    workflow_url: Optional[str] = Field(None, description="CI workflow URL")
    test_suites_executed: List[TestSuiteType] = Field(..., description="Test suites that were executed")
    total_tests: int = Field(0, ge=0, description="Total number of tests")
    passed_tests: int = Field(0, ge=0, description="Number of passed tests")
    failed_tests: int = Field(0, ge=0, description="Number of failed tests")
    skipped_tests: int = Field(0, ge=0, description="Number of skipped tests")
    success_rate: float = Field(0.0, ge=0.0, le=100.0, description="Test success rate percentage")
    coverage_percentage: Optional[float] = Field(None, ge=0.0, le=100.0, description="Code coverage percentage")
    duration_seconds: Optional[int] = Field(None, ge=0, description="Total execution time")
    estimated_duration_minutes: Optional[int] = Field(None, ge=1, description="Estimated execution time for async runs")
    failure_analysis: Optional[FailureAnalysis] = Field(None, description="Analysis of test failures")
    performance_metrics: Optional[Dict[str, Any]] = Field(None, description="Performance test metrics")
    artifacts: List[str] = Field(default=[], description="Generated test artifacts")
    test_suites: Optional[List[TestSuiteResult]] = Field(None, description="Detailed test suite results")
    message: Optional[str] = Field(None, description="Additional status message")


# Conflict Resolution Schemas
class ConflictResolutionRequest(BaseModel):
    """Merge conflict resolution request."""
    pull_request_id: UUID = Field(..., description="Pull request ID with conflicts")
    resolution_strategy: ConflictResolutionStrategy = Field(
        ConflictResolutionStrategy.INTELLIGENT_MERGE,
        description="Conflict resolution strategy to use"
    )
    auto_resolve_enabled: bool = Field(True, description="Enable automatic conflict resolution")
    custom_rules: Optional[Dict[str, Any]] = Field(None, description="Custom resolution rules")


class ConflictFile(BaseModel):
    """Conflicted file information."""
    file_path: str = Field(..., description="Path to conflicted file")
    conflict_type: str = Field(..., description="Type of conflict detected")
    conflict_count: int = Field(..., ge=0, description="Number of conflicts in file")
    resolution_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in resolution")
    auto_resolvable: bool = Field(..., description="Whether conflicts can be automatically resolved")


class ConflictResolutionResponse(BaseResponse):
    """Merge conflict resolution response."""
    pull_request_id: UUID = Field(..., description="Pull request ID")
    resolution_strategy: str = Field(..., description="Strategy used for resolution")
    conflicts_detected: int = Field(..., ge=0, description="Total conflicts detected")
    conflicts_resolved: int = Field(..., ge=0, description="Number of conflicts resolved")
    auto_resolvable: bool = Field(..., description="Whether all conflicts are auto-resolvable")
    resolution_summary: str = Field(..., description="Human-readable resolution summary")
    files_modified: List[str] = Field(..., description="List of files that were modified")
    merge_commit_sha: Optional[str] = Field(None, description="SHA of merge commit if successful")
    resolution_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in resolution")
    manual_intervention_required: bool = Field(..., description="Whether manual intervention is needed")
    resolution_details: Optional[Dict[str, Any]] = Field(None, description="Detailed resolution information")
    conflicted_files: Optional[List[ConflictFile]] = Field(None, description="Details of conflicted files")
    duration_seconds: Optional[int] = Field(None, ge=0, description="Resolution processing time")


# Workflow Automation Schemas
class WorkflowConfig(BaseModel):
    """Workflow execution configuration."""
    stages: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="Stage-specific configurations")
    quality_gates: Optional[Dict[str, Any]] = Field(None, description="Quality gate configurations")
    auto_merge_enabled: bool = Field(False, description="Enable automated merge")
    peer_review_required: bool = Field(False, description="Require peer review")
    notification_settings: Optional[Dict[str, Any]] = Field(None, description="Notification configurations")


class WorkflowExecutionRequest(BaseModel):
    """Workflow execution request."""
    pull_request_id: UUID = Field(..., description="Pull request ID to process")
    trigger: WorkflowTriggerType = Field(..., description="Workflow trigger type")
    workflow_config: Optional[WorkflowConfig] = Field(None, description="Workflow configuration")
    execute_async: bool = Field(True, description="Execute workflow asynchronously")
    timeout_minutes: Optional[int] = Field(120, ge=1, le=600, description="Workflow timeout in minutes")


class QualityGateResult(BaseModel):
    """Quality gate evaluation result."""
    gate_id: str = Field(..., description="Quality gate identifier")
    name: str = Field(..., description="Quality gate name")
    status: str = Field(..., description="Gate status (passed, failed, warning, skipped)")
    score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Gate score if applicable")
    threshold: Optional[Dict[str, Any]] = Field(None, description="Gate threshold configuration")
    details: Optional[Dict[str, Any]] = Field(None, description="Detailed gate results")
    message: Optional[str] = Field(None, description="Gate result message")


class WorkflowExecutionResponse(BaseResponse):
    """Workflow execution response."""
    execution_id: str = Field(..., description="Unique workflow execution ID")
    pull_request_id: UUID = Field(..., description="Pull request ID")
    workflow_id: str = Field(..., description="Workflow template ID")
    status: str = Field(..., description="Current execution status")
    current_stage: WorkflowStageType = Field(..., description="Current workflow stage")
    trigger: str = Field(..., description="Workflow trigger type")
    started_at: str = Field(..., description="Execution start time (ISO format)")
    completed_at: Optional[str] = Field(None, description="Execution completion time (ISO format)")
    duration_seconds: Optional[int] = Field(None, ge=0, description="Total execution time")
    stages_completed: List[str] = Field(default=[], description="List of completed stages")
    stages_failed: List[str] = Field(default=[], description="List of failed stages")
    quality_gates_results: Dict[str, str] = Field(default={}, description="Quality gate results")
    quality_gates_passed: bool = Field(False, description="Whether all quality gates passed")
    automation_summary: Optional[Dict[str, Any]] = Field(None, description="Detailed automation results")
    message: Optional[str] = Field(None, description="Additional status message")
    error_details: Optional[List[ErrorDetail]] = Field(None, description="Error details if execution failed")


# Repository Health Schemas
class RepositoryHealthRequest(BaseModel):
    """Repository health analysis request."""
    include_dependencies: bool = Field(True, description="Include dependency analysis")
    include_security_scan: bool = Field(True, description="Include security vulnerability scan")
    analysis_depth: str = Field("comprehensive", description="Analysis depth (quick, standard, comprehensive)")


class BranchAnalysis(BaseModel):
    """Branch analysis results."""
    branch_name: str = Field(..., description="Branch name")
    commit_count: int = Field(..., ge=0, description="Number of commits")
    commits_behind_main: int = Field(..., ge=0, description="Commits behind main branch")
    last_commit_date: str = Field(..., description="Last commit date (ISO format)")
    author_count: int = Field(..., ge=0, description="Number of unique authors")
    is_stale: bool = Field(..., description="Whether branch is considered stale")
    is_active: bool = Field(..., description="Whether branch is actively developed")


class DependencyAnalysis(BaseModel):
    """Dependency analysis results."""
    total_dependencies: int = Field(..., ge=0, description="Total number of dependencies")
    outdated_dependencies: int = Field(..., ge=0, description="Number of outdated dependencies")
    vulnerable_dependencies: int = Field(..., ge=0, description="Number of vulnerable dependencies")
    critical_vulnerabilities: int = Field(..., ge=0, description="Number of critical vulnerabilities")
    security_risk_score: Optional[float] = Field(None, ge=0.0, le=10.0, description="Security risk score (0-10)")
    recommendations: List[Recommendation] = Field(default=[], description="Update recommendations")
    update_strategy: Optional[Dict[str, Any]] = Field(None, description="Recommended update strategy")


class RepositoryMetrics(BaseModel):
    """Repository metrics."""
    total_commits: int = Field(..., ge=0, description="Total number of commits")
    total_contributors: int = Field(..., ge=0, description="Total number of contributors")
    total_branches: int = Field(..., ge=0, description="Total number of branches")
    active_branches: int = Field(..., ge=0, description="Number of active branches")
    stale_branches: int = Field(..., ge=0, description="Number of stale branches")
    total_files: int = Field(..., ge=0, description="Total number of files")
    code_files: int = Field(..., ge=0, description="Number of code files")
    test_files: int = Field(..., ge=0, description="Number of test files")
    documentation_files: int = Field(..., ge=0, description="Number of documentation files")
    last_activity_days: int = Field(..., ge=0, description="Days since last activity")


class Alert(BaseModel):
    """Repository health alert."""
    type: str = Field(..., description="Alert type")
    severity: str = Field(..., description="Alert severity (critical, warning, info)")
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Alert message")
    action_required: str = Field(..., description="Required action")
    created_at: str = Field(..., description="Alert creation time (ISO format)")


class RepositoryHealthResponse(BaseResponse):
    """Repository health analysis response."""
    repository_id: UUID = Field(..., description="Repository ID")
    overall_health_score: float = Field(..., ge=0.0, le=100.0, description="Overall health score (0-100)")
    health_status: HealthStatus = Field(..., description="Health status classification")
    health_grade: str = Field(..., description="Letter grade (A, B, C, D, F)")
    branch_analysis: Dict[str, BranchAnalysis] = Field(..., description="Analysis of repository branches")
    dependency_analysis: Optional[DependencyAnalysis] = Field(None, description="Dependency analysis results")
    repository_metrics: RepositoryMetrics = Field(..., description="Repository metrics")
    recommendations: List[Recommendation] = Field(..., description="Improvement recommendations")
    alerts: List[Alert] = Field(default=[], description="Critical alerts")
    assessment_duration_seconds: int = Field(..., ge=0, description="Assessment processing time")
    last_assessment_date: str = Field(..., description="Last assessment date (ISO format)")
    trend_analysis: Optional[Dict[str, Any]] = Field(None, description="Health trend analysis")


# Analytics and Reporting Schemas
class PullRequestAnalytics(BaseModel):
    """Pull request analytics."""
    total: int = Field(..., ge=0, description="Total number of pull requests")
    merged: int = Field(..., ge=0, description="Number of merged pull requests")
    closed: int = Field(..., ge=0, description="Number of closed pull requests")
    open: int = Field(..., ge=0, description="Number of open pull requests")
    average_cycle_time_hours: Optional[float] = Field(None, ge=0, description="Average PR cycle time")


class AutomationMetrics(BaseModel):
    """Automation usage metrics."""
    automated_reviews: int = Field(..., ge=0, description="Number of automated reviews")
    automated_merges: int = Field(..., ge=0, description="Number of automated merges")
    conflicts_resolved: int = Field(..., ge=0, description="Number of conflicts resolved")
    quality_gates_passed: int = Field(..., ge=0, description="Number of quality gates passed")
    workflow_executions: int = Field(..., ge=0, description="Number of workflow executions")


class QualityMetrics(BaseModel):
    """Code quality metrics."""
    average_review_score: float = Field(..., ge=0.0, le=1.0, description="Average review score")
    test_success_rate: float = Field(..., ge=0.0, le=1.0, description="Test success rate")
    security_issues_found: int = Field(..., ge=0, description="Security issues found")
    performance_improvements: int = Field(..., ge=0, description="Performance improvements made")
    code_coverage_average: Optional[float] = Field(None, ge=0.0, le=100.0, description="Average code coverage")


class EfficiencyMetrics(BaseModel):
    """Efficiency and productivity metrics."""
    average_pr_cycle_time_hours: float = Field(..., ge=0, description="Average PR cycle time")
    automated_workflow_usage: float = Field(..., ge=0.0, le=1.0, description="Automated workflow usage rate")
    manual_intervention_rate: float = Field(..., ge=0.0, le=1.0, description="Manual intervention rate")
    time_saved_hours: Optional[float] = Field(None, ge=0, description="Estimated time saved by automation")


class SystemHealthStatus(BaseModel):
    """System health status."""
    status: str = Field(..., description="Overall system status")
    timestamp: str = Field(..., description="Status timestamp (ISO format)")
    components: Dict[str, Dict[str, Union[str, int, float]]] = Field(..., description="Component health status")
    metrics: Dict[str, Union[int, float]] = Field(..., description="System metrics")
    uptime_percentage: Optional[float] = Field(None, ge=0.0, le=100.0, description="System uptime percentage")


# Export all schemas
__all__ = [
    # Base schemas
    "BaseResponse",
    "ErrorDetail",
    "Finding",
    "Recommendation",
    
    # Enums
    "ReviewType",
    "TestSuiteType", 
    "ConflictResolutionStrategy",
    "WorkflowTriggerType",
    "WorkflowStageType",
    "HealthStatus",
    
    # Code Review schemas
    "CodeReviewRequest",
    "CodeReviewResponse",
    
    # Testing schemas
    "TestExecutionRequest",
    "TestExecutionResponse",
    "TestResult",
    "TestSuiteResult", 
    "FailureAnalysis",
    
    # Conflict Resolution schemas
    "ConflictResolutionRequest",
    "ConflictResolutionResponse",
    "ConflictFile",
    
    # Workflow Automation schemas
    "WorkflowExecutionRequest",
    "WorkflowExecutionResponse",
    "WorkflowConfig",
    "QualityGateResult",
    
    # Repository Health schemas
    "RepositoryHealthRequest",
    "RepositoryHealthResponse",
    "BranchAnalysis",
    "DependencyAnalysis", 
    "RepositoryMetrics",
    "Alert",
    
    # Analytics schemas
    "PullRequestAnalytics",
    "AutomationMetrics",
    "QualityMetrics",
    "EfficiencyMetrics",
    "SystemHealthStatus"
]