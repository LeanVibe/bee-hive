"""
Enhanced Tool Registry & Discovery System for LeanVibe Agent Hive 2.0

Provides dynamic tool registration, discovery, and validation capabilities enabling
agents to automatically discover and safely use new tools without system restart.

Key Features:
- Dynamic tool registration with input schema validation
- Natural language tool descriptions for AI agent understanding
- Plugin architecture for extensible tool ecosystem
- Comprehensive tool usage analytics and monitoring
- Security validation and access control integration
- Real-time tool availability and health checking
"""

import asyncio
import inspect
import logging
import importlib
import pkg_resources
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Type, Union, Set
from uuid import UUID, uuid4
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, validator
from fastapi import HTTPException

from ..models.agent import Agent
from ..models.task import Task
from ..core.database import get_async_session
from ..core.enhanced_security_safeguards import (
    validate_agent_action, 
    ControlDecision,
    get_enhanced_security_safeguards
)
from .external_tools import (
    GitIntegration, 
    GitHubIntegration, 
    DockerIntegration,
    ToolType,
    OperationStatus
)


logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories for organizing tools."""
    VERSION_CONTROL = "version_control"
    DEPLOYMENT = "deployment"
    TESTING = "testing"
    MONITORING = "monitoring"
    COMMUNICATION = "communication"
    DATA_PROCESSING = "data_processing"
    SECURITY = "security"
    DEVELOPMENT = "development"
    INFRASTRUCTURE = "infrastructure"
    UTILITY = "utility"


class ToolAccessLevel(Enum):
    """Access levels for tool security."""
    PUBLIC = "public"          # All agents can use
    RESTRICTED = "restricted"  # Requires special permissions
    ADMIN_ONLY = "admin_only"  # Only admin agents
    QUARANTINED = "quarantined" # Temporarily disabled


@dataclass
class ToolExecutionResult:
    """Result of tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time_ms: int = 0
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ToolInputSchema(BaseModel):
    """Base schema for tool inputs with validation."""
    
    class Config:
        extra = "forbid"  # Prevent additional fields
        validate_assignment = True


class ToolDefinition(BaseModel):
    """Comprehensive tool definition for registration."""
    
    id: str = Field(..., description="Unique tool identifier")
    name: str = Field(..., description="Human-readable tool name")
    description: str = Field(..., description="Natural language description for AI agents")
    category: ToolCategory = Field(..., description="Tool category for organization")
    
    # Technical specifications
    function: str = Field(..., description="Python function name to call")
    module_path: str = Field(..., description="Module path for the function")
    input_schema: Type[ToolInputSchema] = Field(..., description="Pydantic schema for input validation")
    
    # AI agent context
    usage_examples: List[str] = Field(default_factory=list, description="Example usage patterns")
    when_to_use: str = Field("", description="Guidance on when to use this tool")
    limitations: str = Field("", description="Known limitations and constraints")
    
    # Access control and security
    access_level: ToolAccessLevel = Field(default=ToolAccessLevel.PUBLIC)
    required_permissions: List[str] = Field(default_factory=list)
    risk_level: str = Field(default="low", description="Security risk assessment")
    
    # Performance and monitoring
    timeout_seconds: int = Field(default=30, description="Maximum execution time")
    retry_count: int = Field(default=0, description="Number of retries on failure")
    rate_limit_per_minute: int = Field(default=60, description="Rate limiting")
    
    # Metadata
    version: str = Field(default="1.0.0", description="Tool version")
    author: str = Field("", description="Tool author/maintainer")
    tags: List[str] = Field(default_factory=list, description="Additional tags")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        arbitrary_types_allowed = True  # Allow custom types like ToolCategory


class ToolUsageMetrics(BaseModel):
    """Metrics for tool usage tracking."""
    
    tool_id: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time_ms: float = 0.0
    last_used_at: Optional[datetime] = None
    agents_using_tool: Set[str] = Field(default_factory=set)
    error_patterns: Dict[str, int] = Field(default_factory=dict)
    performance_trends: List[Dict[str, Any]] = Field(default_factory=list)


class ToolHealthStatus(BaseModel):
    """Health status for tools."""
    
    tool_id: str
    is_healthy: bool = True
    last_health_check: datetime = Field(default_factory=datetime.utcnow)
    health_score: float = 1.0  # 0.0 to 1.0
    issues: List[str] = Field(default_factory=list)
    dependencies_status: Dict[str, bool] = Field(default_factory=dict)


class EnhancedToolRegistry:
    """
    Dynamic tool registry with discovery, validation, and monitoring.
    
    Features:
    - Dynamic tool registration and discovery
    - Input validation with Pydantic schemas
    - Security integration with enhanced safeguards
    - Comprehensive usage analytics and health monitoring
    - Plugin architecture for extensible tool ecosystem
    """
    
    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}
        self.tool_functions: Dict[str, Callable] = {}
        self.usage_metrics: Dict[str, ToolUsageMetrics] = {}
        self.health_status: Dict[str, ToolHealthStatus] = {}
        self.plugin_managers: List[Callable] = []
        
        # Rate limiting tracking
        self.rate_limit_tracking: Dict[str, List[datetime]] = {}
        
        # Initialize with core tools
        self._initialize_core_tools()
    
    def register_tool(
        self,
        tool_definition: ToolDefinition,
        tool_function: Callable,
        override: bool = False
    ) -> bool:
        """
        Register a new tool in the registry.
        
        Args:
            tool_definition: Tool definition with metadata
            tool_function: Actual function to execute
            override: Whether to override existing tool
            
        Returns:
            bool: Success status
        """
        try:
            if tool_definition.id in self.tools and not override:
                logger.warning(f"Tool {tool_definition.id} already registered")
                return False
            
            # Validate function signature matches schema
            if not self._validate_function_schema(tool_function, tool_definition.input_schema):
                logger.error(f"Function signature mismatch for tool {tool_definition.id}")
                return False
            
            # Register tool
            self.tools[tool_definition.id] = tool_definition
            self.tool_functions[tool_definition.id] = tool_function
            
            # Initialize metrics and health status
            self.usage_metrics[tool_definition.id] = ToolUsageMetrics(tool_id=tool_definition.id)
            self.health_status[tool_definition.id] = ToolHealthStatus(tool_id=tool_definition.id)
            
            logger.info(f"Tool registered successfully: {tool_definition.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering tool {tool_definition.id}: {e}")
            return False
    
    def unregister_tool(self, tool_id: str) -> bool:
        """Unregister a tool from the registry."""
        try:
            if tool_id not in self.tools:
                return False
            
            del self.tools[tool_id]
            del self.tool_functions[tool_id]
            del self.usage_metrics[tool_id]
            del self.health_status[tool_id]
            
            if tool_id in self.rate_limit_tracking:
                del self.rate_limit_tracking[tool_id]
            
            logger.info(f"Tool unregistered: {tool_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error unregistering tool {tool_id}: {e}")
            return False
    
    def discover_tools(
        self,
        agent_id: Optional[UUID] = None,
        category: Optional[ToolCategory] = None,
        access_level: Optional[ToolAccessLevel] = None,
        search_query: Optional[str] = None
    ) -> List[ToolDefinition]:
        """
        Discover available tools based on filters.
        
        Args:
            agent_id: Agent requesting tools (for permission filtering)
            category: Filter by tool category
            access_level: Filter by access level
            search_query: Search in name/description
            
        Returns:
            List of available tool definitions
        """
        try:
            available_tools = []
            
            for tool_id, tool_def in self.tools.items():
                # Check health status
                if not self.health_status[tool_id].is_healthy:
                    continue
                
                # Filter by category
                if category and tool_def.category != category:
                    continue
                
                # Filter by access level
                if access_level and tool_def.access_level != access_level:
                    continue
                
                # Search query filtering
                if search_query:
                    query_lower = search_query.lower()
                    if (query_lower not in tool_def.name.lower() and 
                        query_lower not in tool_def.description.lower() and
                        not any(query_lower in tag.lower() for tag in tool_def.tags)):
                        continue
                
                # TODO: Add agent permission checking when agent_id provided
                
                available_tools.append(tool_def)
            
            # Sort by category, then by name
            available_tools.sort(key=lambda t: (t.category.value, t.name))
            
            logger.debug(f"Discovered {len(available_tools)} tools for agent {agent_id}")
            return available_tools
            
        except Exception as e:
            logger.error(f"Error discovering tools: {e}")
            return []
    
    async def execute_tool(
        self,
        tool_id: str,
        agent_id: UUID,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolExecutionResult:
        """
        Execute a tool with security validation and monitoring.
        
        Args:
            tool_id: Tool to execute
            agent_id: Agent requesting execution
            input_data: Input parameters
            context: Additional context for execution
            
        Returns:
            ToolExecutionResult with execution details
        """
        start_time = datetime.utcnow()
        
        try:
            # Validate tool exists
            if tool_id not in self.tools:
                return ToolExecutionResult(
                    success=False,
                    output=None,
                    error=f"Tool {tool_id} not found"
                )
            
            tool_def = self.tools[tool_id]
            tool_function = self.tool_functions[tool_id]
            
            # Check health status
            if not self.health_status[tool_id].is_healthy:
                return ToolExecutionResult(
                    success=False,
                    output=None,
                    error=f"Tool {tool_id} is currently unhealthy"
                )
            
            # Rate limiting check
            if not self._check_rate_limit(tool_id, agent_id):
                return ToolExecutionResult(
                    success=False,
                    output=None,
                    error=f"Rate limit exceeded for tool {tool_id}"
                )
            
            # Security validation
            security_decision, security_reason, security_data = await validate_agent_action(
                agent_id=agent_id,
                action_type="tool_execution",
                resource_type="external_tool",
                resource_id=tool_id,
                metadata={
                    "tool_category": tool_def.category.value,
                    "risk_level": tool_def.risk_level,
                    "input_data": input_data
                }
            )
            
            if security_decision != ControlDecision.ALLOW:
                return ToolExecutionResult(
                    success=False,
                    output=None,
                    error=f"Security check failed: {security_reason}"
                )
            
            # Input validation
            try:
                validated_input = tool_def.input_schema(**input_data)
            except ValidationError as e:
                return ToolExecutionResult(
                    success=False,
                    output=None,
                    error=f"Input validation failed: {e}"
                )
            
            # Execute tool with timeout
            try:
                result = await asyncio.wait_for(
                    self._execute_tool_function(tool_function, validated_input.dict(), context),
                    timeout=tool_def.timeout_seconds
                )
                
                execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                
                # Update metrics
                await self._update_tool_metrics(tool_id, agent_id, True, execution_time)
                
                return ToolExecutionResult(
                    success=True,
                    output=result,
                    execution_time_ms=execution_time,
                    metadata={
                        "tool_version": tool_def.version,
                        "security_validated": True
                    }
                )
                
            except asyncio.TimeoutError:
                execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                await self._update_tool_metrics(tool_id, agent_id, False, execution_time, "timeout")
                
                return ToolExecutionResult(
                    success=False,
                    output=None,
                    error=f"Tool execution timed out after {tool_def.timeout_seconds}s",
                    execution_time_ms=execution_time
                )
            
        except Exception as e:
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self._update_tool_metrics(tool_id, agent_id, False, execution_time, str(e))
            
            logger.error(f"Error executing tool {tool_id}: {e}")
            return ToolExecutionResult(
                success=False,
                output=None,
                error=f"Tool execution failed: {str(e)}",
                execution_time_ms=execution_time
            )
    
    async def health_check_tool(self, tool_id: str) -> ToolHealthStatus:
        """Perform health check on a specific tool."""
        try:
            if tool_id not in self.tools:
                return ToolHealthStatus(
                    tool_id=tool_id,
                    is_healthy=False,
                    health_score=0.0,
                    issues=["Tool not found"]
                )
            
            tool_def = self.tools[tool_id]
            tool_function = self.tool_functions[tool_id]
            issues = []
            dependencies_status = {}
            
            # Check if function is callable
            if not callable(tool_function):
                issues.append("Tool function is not callable")
            
            # Check dependencies if specified
            if hasattr(tool_function, '__dependencies__'):
                for dep in tool_function.__dependencies__:
                    try:
                        importlib.import_module(dep)
                        dependencies_status[dep] = True
                    except ImportError:
                        dependencies_status[dep] = False
                        issues.append(f"Missing dependency: {dep}")
            
            # Calculate health score
            health_score = 1.0
            if issues:
                health_score = max(0.0, 1.0 - (len(issues) * 0.2))
            
            health_status = ToolHealthStatus(
                tool_id=tool_id,
                is_healthy=len(issues) == 0,
                health_score=health_score,
                issues=issues,
                dependencies_status=dependencies_status
            )
            
            self.health_status[tool_id] = health_status
            return health_status
            
        except Exception as e:
            logger.error(f"Error checking health for tool {tool_id}: {e}")
            return ToolHealthStatus(
                tool_id=tool_id,
                is_healthy=False,
                health_score=0.0,
                issues=[f"Health check failed: {str(e)}"]
            )
    
    async def health_check_all_tools(self) -> Dict[str, ToolHealthStatus]:
        """Perform health checks on all registered tools."""
        health_results = {}
        
        for tool_id in self.tools.keys():
            health_results[tool_id] = await self.health_check_tool(tool_id)
        
        return health_results
    
    def get_tool_metrics(self, tool_id: str) -> Optional[ToolUsageMetrics]:
        """Get usage metrics for a specific tool."""
        return self.usage_metrics.get(tool_id)
    
    def get_all_metrics(self) -> Dict[str, ToolUsageMetrics]:
        """Get usage metrics for all tools."""
        return self.usage_metrics.copy()
    
    def get_agent_tool_usage(self, agent_id: UUID) -> Dict[str, Any]:
        """Get tool usage summary for a specific agent."""
        agent_str = str(agent_id)
        tool_usage = {}
        
        for tool_id, metrics in self.usage_metrics.items():
            if agent_str in metrics.agents_using_tool:
                tool_usage[tool_id] = {
                    "tool_name": self.tools[tool_id].name,
                    "category": self.tools[tool_id].category.value,
                    "last_used": metrics.last_used_at,
                    "success_rate": (
                        metrics.successful_executions / max(1, metrics.total_executions)
                    )
                }
        
        return tool_usage
    
    def register_plugin_manager(self, plugin_manager: Callable) -> None:
        """Register a plugin manager for automatic tool discovery."""
        self.plugin_managers.append(plugin_manager)
        logger.info("Plugin manager registered")
    
    async def discover_plugins(self) -> List[ToolDefinition]:
        """Discover tools from registered plugin managers."""
        discovered_tools = []
        
        for plugin_manager in self.plugin_managers:
            try:
                tools = await plugin_manager()
                discovered_tools.extend(tools)
            except Exception as e:
                logger.error(f"Error discovering plugins: {e}")
        
        return discovered_tools
    
    def _initialize_core_tools(self) -> None:
        """Initialize core tools from existing integrations."""
        try:
            # Git tools
            self._register_git_tools()
            
            # GitHub tools  
            self._register_github_tools()
            
            # Docker tools
            self._register_docker_tools()
            
            logger.info("Core tools initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing core tools: {e}")
    
    def _register_git_tools(self) -> None:
        """Register Git integration tools."""
        
        class GitCloneSchema(ToolInputSchema):
            repository_url: str = Field(..., description="Git repository URL to clone")
            target_directory: Optional[str] = Field(None, description="Target directory name")
            branch: Optional[str] = Field(None, description="Specific branch to clone")
        
        git_clone_def = ToolDefinition(
            id="git_clone",
            name="Git Clone Repository",
            description="Clone a Git repository into the agent's workspace for version control operations",
            category=ToolCategory.VERSION_CONTROL,
            function="clone_repository",
            module_path="app.core.external_tools.GitIntegration",
            input_schema=GitCloneSchema,
            usage_examples=[
                "Clone the main branch: git_clone(repository_url='https://github.com/user/repo.git')",
                "Clone specific branch: git_clone(repository_url='https://github.com/user/repo.git', branch='develop')"
            ],
            when_to_use="When you need to access source code from a Git repository for development tasks",
            limitations="Requires valid Git repository URL and appropriate access permissions",
            risk_level="moderate",
            timeout_seconds=120  # Git operations can take longer
        )
        
        # Note: In production, this would reference the actual GitIntegration method
        async def git_clone_wrapper(repository_url: str, target_directory: str = None, branch: str = None):
            from .external_tools import external_tools
            return await external_tools.git.clone_repository(
                agent_id="system",  # Would use actual agent_id in practice
                repository_url=repository_url,
                target_directory=target_directory,
                branch=branch
            )
        
        self.register_tool(git_clone_def, git_clone_wrapper)
        
        # Add more Git tools...
        class GitCommitSchema(ToolInputSchema):
            repository_id: str = Field(..., description="Repository ID from previous clone")
            message: str = Field(..., description="Commit message")
            files: Optional[List[str]] = Field(None, description="Specific files to commit")
        
        git_commit_def = ToolDefinition(
            id="git_commit",
            name="Git Commit Changes",
            description="Commit changes to a Git repository with a descriptive message",
            category=ToolCategory.VERSION_CONTROL,
            function="commit_changes",
            module_path="app.core.external_tools.GitIntegration",
            input_schema=GitCommitSchema,
            usage_examples=[
                "Commit all changes: git_commit(repository_id='repo_123', message='Fix bug in user authentication')",
                "Commit specific files: git_commit(repository_id='repo_123', message='Update config', files=['config.py'])"
            ],
            when_to_use="After making changes to code that you want to save in version control",
            limitations="Repository must be cloned first and changes must be made",
            risk_level="low"
        )
        
        async def git_commit_wrapper(repository_id: str, message: str, files: List[str] = None):
            from .external_tools import external_tools
            return await external_tools.git.commit_changes(repository_id, message, files)
        
        self.register_tool(git_commit_def, git_commit_wrapper)
    
    def _register_github_tools(self) -> None:
        """Register GitHub integration tools."""
        
        class GitHubCreatePRSchema(ToolInputSchema):
            repository_id: str = Field(..., description="GitHub repository ID")
            title: str = Field(..., description="Pull request title")
            body: str = Field(..., description="Pull request description")
            head_branch: str = Field(..., description="Source branch for PR")
            base_branch: str = Field(default="main", description="Target branch for PR")
        
        github_pr_def = ToolDefinition(
            id="github_create_pr",
            name="Create GitHub Pull Request",
            description="Create a pull request on GitHub to propose code changes for review",
            category=ToolCategory.VERSION_CONTROL,
            function="create_pull_request",
            module_path="app.core.external_tools.GitHubIntegration",
            input_schema=GitHubCreatePRSchema,
            usage_examples=[
                "Create PR: github_create_pr(repository_id='repo_123', title='Add new feature', body='Description of changes', head_branch='feature-branch')"
            ],
            when_to_use="When you want to propose code changes for review and integration",
            limitations="Requires GitHub access token and repository permissions",
            access_level=ToolAccessLevel.RESTRICTED,
            risk_level="moderate"
        )
        
        async def github_pr_wrapper(repository_id: str, title: str, body: str, head_branch: str, base_branch: str = "main"):
            from .external_tools import external_tools
            return await external_tools.github.create_pull_request(repository_id, title, body, head_branch, base_branch)
        
        self.register_tool(github_pr_def, github_pr_wrapper)
    
    def _register_docker_tools(self) -> None:
        """Register Docker integration tools."""
        
        class DockerBuildSchema(ToolInputSchema):
            dockerfile_path: str = Field(..., description="Path to Dockerfile")
            image_name: str = Field(..., description="Name for the Docker image")
            context_path: str = Field(default=".", description="Build context path")
            build_args: Optional[Dict[str, str]] = Field(None, description="Build arguments")
        
        docker_build_def = ToolDefinition(
            id="docker_build",
            name="Build Docker Image",
            description="Build a Docker image from a Dockerfile for containerized deployment",
            category=ToolCategory.DEPLOYMENT,
            function="build_image",
            module_path="app.core.external_tools.DockerIntegration",
            input_schema=DockerBuildSchema,
            usage_examples=[
                "Build image: docker_build(dockerfile_path='./Dockerfile', image_name='my-app:latest')",
                "Build with args: docker_build(dockerfile_path='./Dockerfile', image_name='my-app:latest', build_args={'ENV': 'production'})"
            ],
            when_to_use="When you need to create a containerized version of an application",
            limitations="Requires Docker installed and valid Dockerfile",
            risk_level="moderate",
            timeout_seconds=300  # Docker builds can take time
        )
        
        async def docker_build_wrapper(dockerfile_path: str, image_name: str, context_path: str = ".", build_args: Dict[str, str] = None):
            from .external_tools import external_tools
            return await external_tools.docker.build_image(
                agent_id="system",  # Would use actual agent_id
                dockerfile_path=dockerfile_path,
                image_name=image_name,
                context_path=context_path,
                build_args=build_args
            )
        
        self.register_tool(docker_build_def, docker_build_wrapper)
    
    def _validate_function_schema(self, function: Callable, schema: Type[ToolInputSchema]) -> bool:
        """Validate that function signature matches expected schema."""
        try:
            sig = inspect.signature(function)
            schema_fields = schema.__fields__
            
            # Check if all required schema fields have corresponding function parameters
            for field_name, field_info in schema_fields.items():
                if field_name not in sig.parameters:
                    logger.error(f"Function missing parameter: {field_name}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating function schema: {e}")
            return False
    
    async def _execute_tool_function(
        self, 
        function: Callable, 
        input_data: Dict[str, Any], 
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Execute tool function with proper async handling."""
        try:
            if asyncio.iscoroutinefunction(function):
                return await function(**input_data)
            else:
                return function(**input_data)
        except Exception as e:
            logger.error(f"Error executing tool function: {e}")
            raise
    
    def _check_rate_limit(self, tool_id: str, agent_id: UUID) -> bool:
        """Check if tool execution is within rate limits."""
        try:
            tool_def = self.tools[tool_id]
            now = datetime.utcnow()
            cutoff = now - timedelta(minutes=1)
            
            key = f"{tool_id}:{agent_id}"
            
            if key not in self.rate_limit_tracking:
                self.rate_limit_tracking[key] = []
            
            # Remove old entries
            self.rate_limit_tracking[key] = [
                timestamp for timestamp in self.rate_limit_tracking[key]
                if timestamp > cutoff
            ]
            
            # Check rate limit
            if len(self.rate_limit_tracking[key]) >= tool_def.rate_limit_per_minute:
                return False
            
            # Add current request
            self.rate_limit_tracking[key].append(now)
            return True
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return True  # Allow execution on error
    
    async def _update_tool_metrics(
        self, 
        tool_id: str, 
        agent_id: UUID, 
        success: bool, 
        execution_time_ms: int,
        error_message: str = None
    ) -> None:
        """Update tool usage metrics."""
        try:
            metrics = self.usage_metrics[tool_id]
            
            metrics.total_executions += 1
            if success:
                metrics.successful_executions += 1
            else:
                metrics.failed_executions += 1
                
                # Track error patterns
                if error_message:
                    error_key = error_message[:50]  # First 50 chars
                    metrics.error_patterns[error_key] = metrics.error_patterns.get(error_key, 0) + 1
            
            # Update average execution time
            total_time = metrics.average_execution_time_ms * (metrics.total_executions - 1) + execution_time_ms
            metrics.average_execution_time_ms = total_time / metrics.total_executions
            
            metrics.last_used_at = datetime.utcnow()
            metrics.agents_using_tool.add(str(agent_id))
            
            # Add performance trend data point
            metrics.performance_trends.append({
                "timestamp": datetime.utcnow().isoformat(),
                "execution_time_ms": execution_time_ms,
                "success": success
            })
            
            # Keep only last 100 trend points
            if len(metrics.performance_trends) > 100:
                metrics.performance_trends = metrics.performance_trends[-100:]
            
        except Exception as e:
            logger.error(f"Error updating tool metrics: {e}")


# Global enhanced tool registry instance
_enhanced_tool_registry: Optional[EnhancedToolRegistry] = None


def get_enhanced_tool_registry() -> EnhancedToolRegistry:
    """Get singleton enhanced tool registry instance."""
    global _enhanced_tool_registry
    if _enhanced_tool_registry is None:
        _enhanced_tool_registry = EnhancedToolRegistry()
    return _enhanced_tool_registry


# Convenience functions for common operations
async def discover_available_tools(
    agent_id: UUID,
    category: Optional[ToolCategory] = None,
    search_query: Optional[str] = None
) -> List[ToolDefinition]:
    """Convenience function for tool discovery."""
    registry = get_enhanced_tool_registry()
    return registry.discover_tools(agent_id=agent_id, category=category, search_query=search_query)


async def execute_tool_by_id(
    tool_id: str,
    agent_id: UUID,
    input_data: Dict[str, Any]
) -> ToolExecutionResult:
    """Convenience function for tool execution."""
    registry = get_enhanced_tool_registry()
    return await registry.execute_tool(tool_id, agent_id, input_data)


async def get_agent_tool_recommendations(
    agent_id: UUID,
    current_task: Optional[str] = None
) -> List[ToolDefinition]:
    """Get tool recommendations for an agent based on current task."""
    registry = get_enhanced_tool_registry()
    
    # Basic recommendation logic - can be enhanced with AI
    if current_task:
        task_lower = current_task.lower()
        
        if any(keyword in task_lower for keyword in ["git", "repository", "clone", "commit"]):
            return registry.discover_tools(agent_id=agent_id, category=ToolCategory.VERSION_CONTROL)
        elif any(keyword in task_lower for keyword in ["docker", "container", "deploy"]):
            return registry.discover_tools(agent_id=agent_id, category=ToolCategory.DEPLOYMENT)
        elif any(keyword in task_lower for keyword in ["test", "testing"]):
            return registry.discover_tools(agent_id=agent_id, category=ToolCategory.TESTING)
    
    # Return most commonly used tools
    all_tools = registry.discover_tools(agent_id=agent_id)
    
    # Sort by usage metrics
    def tool_popularity(tool: ToolDefinition) -> int:
        metrics = registry.get_tool_metrics(tool.id)
        return metrics.total_executions if metrics else 0
    
    return sorted(all_tools, key=tool_popularity, reverse=True)[:10]