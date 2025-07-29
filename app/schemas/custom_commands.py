"""
Pydantic schemas for Custom Commands System API endpoints.

Phase 6.1: Custom Commands System for LeanVibe Agent Hive 2.0
Enables creation and execution of multi-agent workflow commands.
"""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict, validator


class CommandStatus(str, Enum):
    """Command execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class WorkflowStepType(str, Enum):
    """Types of workflow steps."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"


class AgentRole(str, Enum):
    """Standard agent roles for command workflows."""
    BACKEND_ENGINEER = "backend-engineer"
    FRONTEND_BUILDER = "frontend-builder"
    QA_TEST_GUARDIAN = "qa-test-guardian"
    DEVOPS_SPECIALIST = "devops-specialist"
    DATA_ANALYST = "data-analyst"
    SECURITY_AUDITOR = "security-auditor"
    PRODUCT_MANAGER = "product-manager"
    TECHNICAL_WRITER = "technical-writer"


class AgentRequirement(BaseModel):
    """Agent requirement specification for workflow steps."""
    role: AgentRole = Field(..., description="Required agent role")
    specialization: List[str] = Field(default_factory=list, description="Required specializations")
    min_experience_level: int = Field(default=1, ge=1, le=5, description="Minimum experience level (1-5)")
    required_capabilities: List[str] = Field(default_factory=list, description="Required capabilities")
    resource_requirements: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Resource requirements")


class WorkflowStep(BaseModel):
    """Individual workflow step definition."""
    step: str = Field(..., description="Step identifier")
    step_type: WorkflowStepType = Field(default=WorkflowStepType.SEQUENTIAL, description="Step execution type")
    agent: Optional[AgentRole] = Field(None, description="Required agent role for single-agent steps")
    agents: Optional[List[AgentRequirement]] = Field(None, description="Required agents for multi-agent steps")
    task: str = Field(..., description="Task description")
    inputs: List[str] = Field(default_factory=list, description="Input requirements")
    outputs: List[str] = Field(default_factory=list, description="Expected outputs")
    depends_on: List[str] = Field(default_factory=list, description="Step dependencies")
    timeout_minutes: Optional[int] = Field(default=60, ge=1, description="Step timeout in minutes")
    retry_count: int = Field(default=0, ge=0, le=5, description="Maximum retry attempts")
    conditions: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Conditional execution rules")
    parallel: Optional[List['WorkflowStep']] = Field(None, description="Parallel sub-steps")
    
    @validator('parallel')
    def validate_parallel_steps(cls, v, values):
        """Validate parallel steps configuration."""
        if v is not None and values.get('step_type') != WorkflowStepType.PARALLEL:
            raise ValueError("Parallel steps can only be defined for parallel step type")
        return v


class SecurityPolicy(BaseModel):
    """Security policy for command execution."""
    allowed_operations: List[str] = Field(default_factory=list, description="Allowed operations")
    restricted_paths: List[str] = Field(default_factory=list, description="Restricted file system paths")
    network_access: bool = Field(default=False, description="Allow network access")
    resource_limits: Dict[str, Any] = Field(default_factory=dict, description="Resource usage limits")
    audit_level: str = Field(default="standard", description="Audit logging level")


class CommandDefinition(BaseModel):
    """Complete command definition schema."""
    model_config = ConfigDict(extra="forbid")
    
    name: str = Field(..., min_length=1, max_length=100, description="Command name")
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$", description="Semantic version")
    description: str = Field(..., max_length=500, description="Command description")
    category: str = Field(default="general", description="Command category")
    tags: List[str] = Field(default_factory=list, description="Command tags")
    
    # Agent requirements
    agents: List[AgentRequirement] = Field(..., min_items=1, description="Required agents")
    
    # Workflow definition
    workflow: List[WorkflowStep] = Field(..., min_items=1, description="Workflow steps")
    
    # Configuration
    default_timeout_minutes: int = Field(default=120, ge=1, description="Default command timeout")
    max_parallel_tasks: int = Field(default=10, ge=1, le=50, description="Maximum parallel tasks")
    failure_strategy: str = Field(default="fail_fast", description="Failure handling strategy")
    
    # Security
    security_policy: SecurityPolicy = Field(default_factory=SecurityPolicy, description="Security policy")
    requires_approval: bool = Field(default=False, description="Requires human approval")
    
    # Metadata
    author: Optional[str] = Field(None, description="Command author")
    documentation_url: Optional[str] = Field(None, description="Documentation URL")
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Creation timestamp")


class CommandCreateRequest(BaseModel):
    """Request schema for creating new commands."""
    definition: CommandDefinition = Field(..., description="Command definition")
    validate_agents: bool = Field(default=True, description="Validate agent availability")
    dry_run: bool = Field(default=False, description="Perform dry run validation only")


class CommandUpdateRequest(BaseModel):
    """Request schema for updating existing commands."""
    definition: Optional[CommandDefinition] = Field(None, description="Updated command definition")
    enabled: Optional[bool] = Field(None, description="Enable/disable command")
    version_increment: str = Field(default="patch", description="Version increment type")


class CommandExecutionRequest(BaseModel):
    """Request schema for executing commands."""
    command_name: str = Field(..., description="Command name to execute")
    command_version: Optional[str] = Field(None, description="Specific command version")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Execution parameters")
    context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")
    priority: str = Field(default="medium", description="Execution priority")
    timeout_override: Optional[int] = Field(None, ge=1, description="Override default timeout")
    agent_preferences: Optional[Dict[str, str]] = Field(None, description="Preferred agent assignments")


class StepExecutionResult(BaseModel):
    """Result of individual workflow step execution."""
    step_id: str = Field(..., description="Step identifier")
    status: CommandStatus = Field(..., description="Step execution status")
    agent_id: Optional[str] = Field(None, description="Executing agent ID")
    start_time: Optional[datetime] = Field(None, description="Step start time")
    end_time: Optional[datetime] = Field(None, description="Step end time")
    execution_time_seconds: Optional[float] = Field(None, description="Execution time in seconds")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Step outputs")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    retry_count: int = Field(default=0, description="Number of retries attempted")


class CommandExecutionResult(BaseModel):
    """Result of complete command execution."""
    execution_id: uuid.UUID = Field(..., description="Unique execution ID")
    command_name: str = Field(..., description="Executed command name")
    command_version: str = Field(..., description="Executed command version")
    status: CommandStatus = Field(..., description="Overall execution status")
    
    # Timing
    start_time: datetime = Field(..., description="Execution start time")
    end_time: Optional[datetime] = Field(None, description="Execution end time")
    total_execution_time_seconds: Optional[float] = Field(None, description="Total execution time")
    
    # Results
    step_results: List[StepExecutionResult] = Field(default_factory=list, description="Individual step results")
    final_outputs: Dict[str, Any] = Field(default_factory=dict, description="Final command outputs")
    
    # Statistics
    total_steps: int = Field(..., description="Total number of steps")
    completed_steps: int = Field(default=0, description="Number of completed steps")
    failed_steps: int = Field(default=0, description="Number of failed steps")
    
    # Error handling
    error_message: Optional[str] = Field(None, description="Overall error message")
    failure_point: Optional[str] = Field(None, description="Step where failure occurred")


class CommandListResponse(BaseModel):
    """Response schema for command listing."""
    commands: List[Dict[str, Any]] = Field(..., description="List of available commands")
    total: int = Field(..., description="Total number of commands")
    categories: List[str] = Field(default_factory=list, description="Available categories")
    tags: List[str] = Field(default_factory=list, description="Available tags")


class CommandValidationResult(BaseModel):
    """Result of command validation."""
    is_valid: bool = Field(..., description="Whether command is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    agent_availability: Dict[str, bool] = Field(default_factory=dict, description="Agent availability status")
    estimated_duration: Optional[int] = Field(None, description="Estimated execution duration")


class CommandMetrics(BaseModel):
    """Command execution metrics."""
    command_name: str = Field(..., description="Command name")
    total_executions: int = Field(default=0, description="Total number of executions")
    successful_executions: int = Field(default=0, description="Number of successful executions")
    failed_executions: int = Field(default=0, description="Number of failed executions")
    average_execution_time: float = Field(default=0.0, description="Average execution time in seconds")
    success_rate: float = Field(default=0.0, description="Success rate percentage")
    last_executed: Optional[datetime] = Field(None, description="Last execution timestamp")


class CommandStatusResponse(BaseModel):
    """Response schema for command execution status."""
    execution_id: uuid.UUID = Field(..., description="Execution ID")
    status: CommandStatus = Field(..., description="Current execution status")
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0, description="Execution progress percentage")
    current_step: Optional[str] = Field(None, description="Currently executing step")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    logs: List[str] = Field(default_factory=list, description="Recent execution logs")


# Allow forward references for recursive models
WorkflowStep.model_rebuild()