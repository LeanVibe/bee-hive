"""
Universal Agent Interface for Multi-CLI Coordination

This module defines the universal interface that all CLI agents must implement
to participate in coordinated workflows. It provides standardized communication,
task execution, and capability reporting for heterogeneous agent coordination.

Key Components:
- UniversalAgentInterface: Abstract base class for all agents
- AgentTask: Standardized task definition
- AgentResult: Standardized result format
- ExecutionContext: Context preservation and handoff
- AgentCapability: Capability discovery and routing
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import asyncio

# ================================================================================
# Core Enums and Types
# ================================================================================

class AgentType(str, Enum):
    """Supported CLI agent types"""
    CLAUDE_CODE = "claude_code"
    CURSOR = "cursor"
    GEMINI_CLI = "gemini_cli"
    OPENCODE = "opencode"
    GITHUB_COPILOT = "github_copilot"
    PYTHON_AGENT = "python_agent"  # Legacy support
    
class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class CapabilityType(str, Enum):
    """Agent capability categories"""
    CODE_ANALYSIS = "code_analysis"
    CODE_IMPLEMENTATION = "code_implementation"
    CODE_REVIEW = "code_review"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    ARCHITECTURE_DESIGN = "architecture_design"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    SECURITY_ANALYSIS = "security_analysis"
    UI_DEVELOPMENT = "ui_development"
    API_DEVELOPMENT = "api_development"
    DATABASE_DESIGN = "database_design"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"

class HealthState(str, Enum):
    """Agent health states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"

# ================================================================================
# Data Models
# ================================================================================

@dataclass
class AgentCapability:
    """Represents a capability that an agent can perform"""
    type: CapabilityType
    confidence: float  # 0.0 to 1.0
    performance_score: float  # 0.0 to 1.0
    estimated_time_seconds: Optional[int] = None
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    prerequisites: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate capability parameters"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not 0.0 <= self.performance_score <= 1.0:
            raise ValueError("Performance score must be between 0.0 and 1.0")

@dataclass
class ExecutionContext:
    """Context information for agent execution and handoff"""
    worktree_path: str
    git_branch: str
    git_commit_hash: Optional[str] = None
    file_scope: List[str] = field(default_factory=list)
    excluded_paths: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    previous_results: List['AgentResult'] = field(default_factory=list)
    task_chain: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    security_constraints: Dict[str, Any] = field(default_factory=dict)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    
    def add_previous_result(self, result: 'AgentResult') -> None:
        """Add a previous result to the context"""
        self.previous_results.append(result)
        self.task_chain.append(result.task_id)
    
    def get_relevant_files(self, task_type: CapabilityType) -> List[str]:
        """Get files relevant to a specific task type"""
        # Implementation would filter files based on task type
        return self.file_scope
    
    def validate_path_access(self, path: str) -> bool:
        """Validate that a path is accessible within security constraints"""
        # Implementation would check path permissions and constraints
        return True  # Placeholder

@dataclass
class AgentTask:
    """Standardized task definition for agent execution"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: CapabilityType = CapabilityType.CODE_ANALYSIS
    title: str = ""
    description: str = ""
    requirements: List[str] = field(default_factory=list)
    input_data: Dict[str, Any] = field(default_factory=dict)
    expected_outputs: List[str] = field(default_factory=list)
    context: Optional[ExecutionContext] = None
    priority: int = 5  # 1 (highest) to 10 (lowest)
    timeout_seconds: int = 300  # 5 minutes default
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate task parameters"""
        if not 1 <= self.priority <= 10:
            raise ValueError("Priority must be between 1 and 10")
        if self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")
        if not self.title and not self.description:
            raise ValueError("Task must have either title or description")

@dataclass
class AgentResult:
    """Standardized result format for agent task execution"""
    task_id: str
    agent_id: str
    agent_type: AgentType
    status: TaskStatus
    output_data: Dict[str, Any] = field(default_factory=dict)
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    files_deleted: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    error_details: Dict[str, Any] = field(default_factory=dict)
    execution_time_seconds: float = 0.0
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def mark_started(self) -> None:
        """Mark the task as started"""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.utcnow()
    
    def mark_completed(self, success: bool = True) -> None:
        """Mark the task as completed"""
        self.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
        self.completed_at = datetime.utcnow()
        if self.started_at:
            self.execution_time_seconds = (self.completed_at - self.started_at).total_seconds()
    
    def add_quality_metric(self, metric_name: str, value: float) -> None:
        """Add a quality metric to the result"""
        if not 0.0 <= value <= 1.0:
            raise ValueError("Quality metrics must be between 0.0 and 1.0")
        self.quality_metrics[metric_name] = value

@dataclass
class HealthStatus:
    """Agent health and status information"""
    agent_id: str
    agent_type: AgentType
    state: HealthState
    response_time_ms: float
    cpu_usage_percent: float
    memory_usage_mb: float
    active_tasks: int
    completed_tasks: int
    failed_tasks: int
    last_activity: datetime
    error_rate: float  # 0.0 to 1.0
    throughput_tasks_per_minute: float
    capabilities_status: Dict[CapabilityType, bool] = field(default_factory=dict)
    last_error: Optional[str] = None
    uptime_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_available(self) -> bool:
        """Check if agent is available for new tasks"""
        return self.state in [HealthState.HEALTHY, HealthState.DEGRADED]
    
    def calculate_load_score(self) -> float:
        """Calculate agent load score (0.0 = no load, 1.0 = fully loaded)"""
        cpu_score = self.cpu_usage_percent / 100.0
        memory_score = min(self.memory_usage_mb / 1024.0, 1.0)  # Assume 1GB baseline
        task_score = min(self.active_tasks / 10.0, 1.0)  # Assume 10 tasks baseline
        return (cpu_score + memory_score + task_score) / 3.0

# ================================================================================
# Universal Agent Interface
# ================================================================================

class UniversalAgentInterface(ABC):
    """
    Universal interface for all CLI agents participating in coordinated workflows.
    
    This abstract base class defines the contract that all agent adapters must
    implement to enable seamless multi-CLI coordination. It provides standardized
    methods for task execution, capability reporting, health monitoring, and
    lifecycle management.
    
    Implementation Requirements:
    - All methods must be implemented by concrete agent adapters
    - Task execution must be isolated within provided context
    - Error handling must be robust with appropriate logging
    - Performance metrics must be tracked and reported
    - Security constraints must be respected
    
    Example Implementation:
        class ClaudeCodeAdapter(UniversalAgentInterface):
            async def execute_task(self, task: AgentTask) -> AgentResult:
                # Translate task to Claude Code format
                # Execute via subprocess with isolation
                # Translate response back to universal format
                
    Thread Safety:
    - All methods should be thread-safe
    - Concurrent task execution should be supported
    - Resource access should be properly synchronized
    """
    
    def __init__(self, agent_id: str, agent_type: AgentType):
        """
        Initialize the universal agent interface.
        
        Args:
            agent_id: Unique identifier for this agent instance
            agent_type: Type of CLI agent this adapter represents
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self._start_time = datetime.utcnow()
        self._task_history: List[AgentResult] = []
        self._current_tasks: Dict[str, AgentTask] = {}
        
    # ================================================================================
    # Core Task Execution
    # ================================================================================
    
    @abstractmethod
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """
        Execute a task using the underlying CLI agent.
        
        This is the core method that adapts universal tasks to agent-specific
        formats, executes them with proper isolation, and returns standardized
        results.
        
        Args:
            task: The standardized task to execute
            
        Returns:
            AgentResult: Standardized execution result
            
        Raises:
            TaskExecutionError: If task execution fails
            SecurityError: If task violates security constraints
            TimeoutError: If task exceeds timeout limit
            
        Implementation Requirements:
        - Validate task against agent capabilities
        - Enforce security constraints from context
        - Execute with proper isolation (worktree, resource limits)
        - Track execution metrics and resource usage
        - Handle errors gracefully with detailed error information
        - Return standardized result format
        
        Example:
            result = await agent.execute_task(AgentTask(
                type=CapabilityType.CODE_ANALYSIS,
                description="Analyze function complexity",
                context=ExecutionContext(worktree_path="/tmp/work")
            ))
        """
        pass
    
    @abstractmethod
    async def get_capabilities(self) -> List[AgentCapability]:
        """
        Report the capabilities this agent can perform.
        
        Returns a list of capabilities with confidence scores and performance
        estimates. This information is used by the orchestrator for intelligent
        task routing.
        
        Returns:
            List[AgentCapability]: Agent's capabilities with metadata
            
        Implementation Requirements:
        - Return accurate capability assessments
        - Include realistic confidence and performance scores
        - Update estimates based on historical performance
        - Consider current agent load and health state
        
        Example:
            capabilities = await agent.get_capabilities()
            for cap in capabilities:
                print(f"{cap.type}: {cap.confidence} confidence")
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """
        Perform comprehensive health check and return status.
        
        Evaluates agent health, performance metrics, resource usage, and
        availability for new tasks. Used by orchestrator for load balancing
        and failure detection.
        
        Returns:
            HealthStatus: Comprehensive health and performance information
            
        Implementation Requirements:
        - Check underlying CLI tool availability
        - Measure response times and resource usage
        - Evaluate error rates and performance trends
        - Determine current capacity and availability
        - Include diagnostic information for troubleshooting
        
        Example:
            health = await agent.health_check()
            if health.is_available():
                print(f"Agent ready: {health.state}")
        """
        pass
    
    # ================================================================================
    # Lifecycle Management
    # ================================================================================
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the agent with configuration.
        
        Performs one-time setup including CLI tool validation, configuration
        loading, resource allocation, and readiness verification.
        
        Args:
            config: Agent-specific configuration parameters
            
        Returns:
            bool: True if initialization successful, False otherwise
            
        Implementation Requirements:
        - Validate CLI tool availability and version
        - Load and validate configuration parameters
        - Allocate necessary resources
        - Establish connections to required services
        - Verify agent is ready for task execution
        
        Example:
            success = await agent.initialize({
                "cli_path": "/usr/local/bin/claude",
                "timeout": 300,
                "max_concurrent_tasks": 3
            })
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """
        Gracefully shutdown the agent.
        
        Performs cleanup including task completion, resource deallocation,
        and state persistence for recovery.
        
        Implementation Requirements:
        - Complete or cancel all active tasks
        - Clean up temporary resources and files
        - Persist state for potential recovery
        - Close connections and release resources
        - Ensure clean shutdown without data loss
        
        Example:
            await agent.shutdown()
        """
        pass
    
    # ================================================================================
    # Task Management
    # ================================================================================
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            bool: True if task was cancelled successfully
        """
        if task_id in self._current_tasks:
            # Implementation should cancel the actual task execution
            del self._current_tasks[task_id]
            return True
        return False
    
    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        Get the status of a specific task.
        
        Args:
            task_id: ID of the task to check
            
        Returns:
            TaskStatus: Current status of the task, or None if not found
        """
        if task_id in self._current_tasks:
            return TaskStatus.IN_PROGRESS
        
        # Check task history
        for result in self._task_history:
            if result.task_id == task_id:
                return result.status
        
        return None
    
    async def list_active_tasks(self) -> List[str]:
        """
        List all currently active task IDs.
        
        Returns:
            List[str]: List of active task IDs
        """
        return list(self._current_tasks.keys())
    
    # ================================================================================
    # Performance and Monitoring
    # ================================================================================
    
    async def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get current performance metrics.
        
        Returns:
            Dict[str, float]: Performance metrics and statistics
        """
        total_tasks = len(self._task_history)
        if total_tasks == 0:
            return {
                "avg_execution_time": 0.0,
                "success_rate": 1.0,
                "throughput": 0.0,
                "error_rate": 0.0
            }
        
        successful_tasks = sum(1 for r in self._task_history if r.status == TaskStatus.COMPLETED)
        total_time = sum(r.execution_time_seconds for r in self._task_history)
        uptime = (datetime.utcnow() - self._start_time).total_seconds()
        
        return {
            "avg_execution_time": total_time / total_tasks if total_tasks > 0 else 0.0,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 1.0,
            "throughput": total_tasks / (uptime / 60.0) if uptime > 0 else 0.0,  # tasks per minute
            "error_rate": (total_tasks - successful_tasks) / total_tasks if total_tasks > 0 else 0.0,
            "uptime_seconds": uptime,
            "total_tasks": total_tasks,
            "active_tasks": len(self._current_tasks)
        }
    
    async def reset_metrics(self) -> None:
        """Reset performance metrics and task history."""
        self._task_history.clear()
        self._start_time = datetime.utcnow()
    
    # ================================================================================
    # Utility Methods
    # ================================================================================
    
    def _add_task_to_history(self, result: AgentResult) -> None:
        """Add a completed task result to history."""
        self._task_history.append(result)
        if result.task_id in self._current_tasks:
            del self._current_tasks[result.task_id]
    
    def _validate_task(self, task: AgentTask) -> None:
        """Validate task parameters and constraints."""
        if not task.context:
            raise ValueError("Task must include execution context")
        
        if not task.context.worktree_path:
            raise ValueError("Task context must specify worktree path")
        
        # Additional validation can be added here
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.agent_type.value}:{self.agent_id}"
    
    def __repr__(self) -> str:
        """Detailed representation of the agent."""
        return (f"UniversalAgentInterface(agent_id='{self.agent_id}', "
                f"agent_type={self.agent_type}, "
                f"active_tasks={len(self._current_tasks)})")

# ================================================================================
# Exceptions
# ================================================================================

class AgentError(Exception):
    """Base exception for agent-related errors."""
    pass

class TaskExecutionError(AgentError):
    """Raised when task execution fails."""
    pass

class SecurityError(AgentError):
    """Raised when task violates security constraints."""
    pass

class ConfigurationError(AgentError):
    """Raised when agent configuration is invalid."""
    pass

class ResourceLimitError(AgentError):
    """Raised when task exceeds resource limits."""
    pass

# ================================================================================
# Utility Functions
# ================================================================================

def create_task(
    task_type: CapabilityType,
    title: str,
    description: str = "",
    context: Optional[ExecutionContext] = None,
    **kwargs
) -> AgentTask:
    """
    Convenience function to create a standardized task.
    
    Args:
        task_type: Type of task to create
        title: Task title
        description: Detailed task description
        context: Execution context (optional)
        **kwargs: Additional task parameters
        
    Returns:
        AgentTask: Configured task ready for execution
    """
    return AgentTask(
        type=task_type,
        title=title,
        description=description,
        context=context,
        **kwargs
    )

def create_execution_context(
    worktree_path: str,
    git_branch: str = "main",
    file_scope: Optional[List[str]] = None,
    **kwargs
) -> ExecutionContext:
    """
    Convenience function to create execution context.
    
    Args:
        worktree_path: Path to isolated worktree
        git_branch: Git branch to use
        file_scope: List of files to work with
        **kwargs: Additional context parameters
        
    Returns:
        ExecutionContext: Configured execution context
    """
    return ExecutionContext(
        worktree_path=worktree_path,
        git_branch=git_branch,
        file_scope=file_scope or [],
        **kwargs
    )

# ================================================================================
# Constants and Configuration
# ================================================================================

# Default timeout values (seconds)
DEFAULT_TASK_TIMEOUT = 300  # 5 minutes
DEFAULT_HEALTH_CHECK_TIMEOUT = 30  # 30 seconds
DEFAULT_INITIALIZATION_TIMEOUT = 60  # 1 minute

# Default resource limits
DEFAULT_RESOURCE_LIMITS = {
    "max_cpu_percent": 80.0,
    "max_memory_mb": 1024.0,
    "max_disk_mb": 10240.0,  # 10GB
    "max_execution_time": 3600  # 1 hour
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "response_time_warning": 1000,  # ms
    "response_time_critical": 5000,  # ms
    "error_rate_warning": 0.05,  # 5%
    "error_rate_critical": 0.15,  # 15%
    "cpu_usage_warning": 70.0,  # %
    "cpu_usage_critical": 90.0   # %
}