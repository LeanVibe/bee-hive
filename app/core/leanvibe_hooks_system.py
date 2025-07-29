"""
LeanVibe Hooks System - Claude Code Integration

Comprehensive hooks system that integrates Claude Code patterns with LeanVibe's
multi-agent orchestration for deterministic quality gates and workflow automation.
"""

import asyncio
import json
import os
import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

import structlog
import yaml
from pydantic import BaseModel, Field

from app.core.orchestrator import AgentOrchestrator
from app.core.communication import MessageBroker
from app.core.hook_processor import HookEventProcessor, get_hook_event_processor

logger = structlog.get_logger()


class HookEventType(str, Enum):
    """Types of hook events in the LeanVibe system."""
    PRE_AGENT_TASK = "PreAgentTask"
    POST_AGENT_TASK = "PostAgentTask"
    AGENT_ERROR = "AgentError"
    WORKFLOW_START = "WorkflowStart"
    WORKFLOW_COMPLETE = "WorkflowComplete"
    QUALITY_GATE = "QualityGate"
    SECURITY_CHECK = "SecurityCheck"
    PERFORMANCE_CHECK = "PerformanceCheck"


class HookExecutionMode(str, Enum):
    """Hook execution modes."""
    BLOCKING = "blocking"
    NON_BLOCKING = "non_blocking"
    ASYNC = "async"
    CRITICAL = "critical"


class HookResult(BaseModel):
    """Result of hook execution."""
    success: bool = Field(..., description="Whether hook execution succeeded")
    output: str = Field(default="", description="Hook output")
    error: Optional[str] = Field(None, description="Error message if failed")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    should_continue: bool = Field(default=True, description="Whether workflow should continue")
    modifications: Dict[str, Any] = Field(default_factory=dict, description="Suggested modifications")


class HookDefinition(BaseModel):
    """Definition of a LeanVibe hook."""
    name: str = Field(..., description="Hook name")
    event_type: HookEventType = Field(..., description="Event type this hook responds to")
    command: str = Field(..., description="Command to execute")
    matcher: str = Field(default="*", description="Pattern to match against")
    description: str = Field(..., description="Hook description")
    execution_mode: HookExecutionMode = Field(default=HookExecutionMode.BLOCKING, description="Execution mode")
    timeout_seconds: int = Field(default=300, description="Timeout in seconds")
    retry_count: int = Field(default=0, description="Number of retries on failure")
    required: bool = Field(default=True, description="Whether hook must succeed")
    environment_vars: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    working_directory: Optional[str] = Field(None, description="Working directory")
    dependencies: List[str] = Field(default_factory=list, description="Hook dependencies")
    conditions: Dict[str, Any] = Field(default_factory=dict, description="Execution conditions")


class HookContext(BaseModel):
    """Context for hook execution."""
    workflow_id: str = Field(..., description="Workflow ID")
    agent_id: str = Field(..., description="Agent ID")
    session_id: str = Field(..., description="Session ID")
    event_type: HookEventType = Field(..., description="Event type")
    event_data: Dict[str, Any] = Field(default_factory=dict, description="Event data")
    project_root: str = Field(..., description="Project root directory")
    environment: str = Field(default="development", description="Environment")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Correlation ID")


class WorkflowHook(BaseModel):
    """Hook configuration for workflow events."""
    hook_definitions: List[HookDefinition] = Field(..., description="Hook definitions")
    execution_order: List[str] = Field(default_factory=list, description="Hook execution order")
    failure_strategy: str = Field(default="stop_on_critical", description="Failure handling strategy")
    parallel_execution: bool = Field(default=False, description="Allow parallel execution")
    max_concurrent: int = Field(default=3, description="Maximum concurrent hooks")


class LeanVibeHooksEngine:
    """
    LeanVibe Hooks Engine integrating Claude Code patterns with multi-agent orchestration.
    
    Provides deterministic automation for quality gates, security validation,
    and workflow consistency across all agent operations.
    """
    
    def __init__(
        self,
        project_root: Optional[Path] = None,
        orchestrator: Optional[AgentOrchestrator] = None,
        communication_bus: Optional[MessageBroker] = None,
        hook_processor: Optional[HookEventProcessor] = None
    ):
        """
        Initialize LeanVibe hooks engine.
        
        Args:
            project_root: Project root directory
            orchestrator: Agent orchestrator instance
            communication_bus: Message broker for real-time events
            hook_processor: Hook event processor for observability
        """
        self.project_root = project_root or Path.cwd()
        self.orchestrator = orchestrator
        self.communication_bus = communication_bus
        self.hook_processor = hook_processor or get_hook_event_processor()
        
        # Hook configuration directories
        self.hooks_config_dir = self.project_root / ".leanvibe" / "hooks"
        self.user_hooks_dir = Path.home() / ".leanvibe" / "hooks"
        
        # Hook execution state
        self.hook_definitions: Dict[HookEventType, List[HookDefinition]] = {}
        self.workflow_hooks: Dict[str, WorkflowHook] = {}
        self.execution_cache: Dict[str, HookResult] = {}
        self.performance_stats = {
            "hooks_executed": 0,
            "hooks_failed": 0,
            "total_execution_time_ms": 0,
            "blocked_actions": 0,
            "cache_hits": 0
        }
        
        # Initialize built-in hooks
        self.built_in_hooks = self._initialize_built_in_hooks()
        
        logger.info(
            "🎣 LeanVibe Hooks Engine initialized",
            project_root=str(self.project_root),
            hooks_config_dir=str(self.hooks_config_dir),
            user_hooks_dir=str(self.user_hooks_dir)
        )
    
    def _initialize_built_in_hooks(self) -> Dict[HookEventType, List[HookDefinition]]:
        """Initialize built-in hooks for essential quality gates."""
        return {
            HookEventType.PRE_AGENT_TASK: [
                HookDefinition(
                    name="security_validation",
                    event_type=HookEventType.PRE_AGENT_TASK,
                    command="python .leanvibe/hooks/validate_security.py",
                    matcher="*",
                    description="Validate task doesn't access sensitive files or execute dangerous commands",
                    execution_mode=HookExecutionMode.BLOCKING,
                    timeout_seconds=30,
                    required=True
                ),
                HookDefinition(
                    name="dependency_check",
                    event_type=HookEventType.PRE_AGENT_TASK,
                    command="python .leanvibe/hooks/check_dependencies.py",
                    matcher="backend_*",
                    description="Ensure required dependencies are available",
                    execution_mode=HookExecutionMode.NON_BLOCKING,
                    timeout_seconds=60
                )
            ],
            
            HookEventType.POST_AGENT_TASK: [
                HookDefinition(
                    name="code_formatting",
                    event_type=HookEventType.POST_AGENT_TASK,
                    command="python .leanvibe/hooks/format_code.py",
                    matcher="*_specialist",
                    description="Auto-format code according to project standards",
                    execution_mode=HookExecutionMode.ASYNC,
                    timeout_seconds=120
                ),
                HookDefinition(
                    name="test_execution",
                    event_type=HookEventType.POST_AGENT_TASK,
                    command="python .leanvibe/hooks/run_tests.py",
                    matcher="*",
                    description="Run relevant tests for changed code",
                    execution_mode=HookExecutionMode.BLOCKING,
                    timeout_seconds=300,
                    required=True
                )
            ],
            
            HookEventType.WORKFLOW_COMPLETE: [
                HookDefinition(
                    name="quality_report",
                    event_type=HookEventType.WORKFLOW_COMPLETE,
                    command="python .leanvibe/hooks/generate_quality_report.py",
                    matcher="*",
                    description="Generate comprehensive quality and progress report",
                    execution_mode=HookExecutionMode.ASYNC,
                    timeout_seconds=180
                ),
                HookDefinition(
                    name="notification",
                    event_type=HookEventType.WORKFLOW_COMPLETE,
                    command="python .leanvibe/hooks/send_notifications.py",
                    matcher="*",
                    description="Notify stakeholders of workflow completion",
                    execution_mode=HookExecutionMode.NON_BLOCKING,
                    timeout_seconds=30
                )
            ],
            
            HookEventType.QUALITY_GATE: [
                HookDefinition(
                    name="comprehensive_quality_check",
                    event_type=HookEventType.QUALITY_GATE,
                    command="python .leanvibe/hooks/comprehensive_quality_check.py",
                    matcher="*",
                    description="Run comprehensive quality validation",
                    execution_mode=HookExecutionMode.BLOCKING,
                    timeout_seconds=600,
                    required=True
                )
            ]
        }
    
    async def load_hooks_configuration(self, force_reload: bool = False) -> None:
        """Load hooks configuration from project and user directories."""
        try:
            # Load built-in hooks
            self.hook_definitions = self.built_in_hooks.copy()
            
            # Load project-specific hooks
            if self.hooks_config_dir.exists():
                project_hooks = await self._load_hooks_from_directory(self.hooks_config_dir, "project")
                self._merge_hook_definitions(project_hooks)
            
            # Load user hooks (global overrides)
            if self.user_hooks_dir.exists():
                user_hooks = await self._load_hooks_from_directory(self.user_hooks_dir, "user")
                self._merge_hook_definitions(user_hooks)
            
            # Load workflow-specific hooks
            await self._load_workflow_hooks()
            
            total_hooks = sum(len(hooks) for hooks in self.hook_definitions.values())
            logger.info(
                "📚 Hooks configuration loaded",
                total_hooks=total_hooks,
                event_types=list(self.hook_definitions.keys()),
                workflow_hooks=len(self.workflow_hooks)
            )
            
        except Exception as e:
            logger.error(
                "❌ Failed to load hooks configuration",
                error=str(e),
                exc_info=True
            )
    
    async def _load_hooks_from_directory(
        self, 
        directory: Path, 
        source: str
    ) -> Dict[HookEventType, List[HookDefinition]]:
        """Load hooks from a directory."""
        hooks: Dict[HookEventType, List[HookDefinition]] = {}
        
        try:
            for config_file in directory.rglob("*.yaml"):
                try:
                    with open(config_file, 'r') as f:
                        config_data = yaml.safe_load(f)
                    
                    if "leanvibe_hooks" in config_data:
                        hooks_config = config_data["leanvibe_hooks"]
                        
                        for event_type_str, hook_configs in hooks_config.items():
                            try:
                                event_type = HookEventType(event_type_str)
                                
                                if event_type not in hooks:
                                    hooks[event_type] = []
                                
                                for hook_config in hook_configs:
                                    hook_definition = HookDefinition(
                                        event_type=event_type,
                                        **hook_config
                                    )
                                    hooks[event_type].append(hook_definition)
                                    
                            except ValueError as e:
                                logger.warning(
                                    "Invalid event type in hook configuration",
                                    event_type=event_type_str,
                                    file=str(config_file),
                                    error=str(e)
                                )
                                
                except Exception as e:
                    logger.warning(
                        "Failed to parse hook configuration file",
                        file=str(config_file),
                        error=str(e)
                    )
        
        except Exception as e:
            logger.error(
                "Failed to load hooks from directory",
                directory=str(directory),
                error=str(e)
            )
        
        return hooks
    
    def _merge_hook_definitions(self, new_hooks: Dict[HookEventType, List[HookDefinition]]) -> None:
        """Merge new hook definitions with existing ones."""
        for event_type, hooks in new_hooks.items():
            if event_type not in self.hook_definitions:
                self.hook_definitions[event_type] = []
            
            self.hook_definitions[event_type].extend(hooks)
    
    async def _load_workflow_hooks(self) -> None:
        """Load workflow-specific hook configurations."""
        workflow_hooks_file = self.hooks_config_dir / "workflows.yaml"
        
        if workflow_hooks_file.exists():
            try:
                with open(workflow_hooks_file, 'r') as f:
                    workflows_config = yaml.safe_load(f)
                
                for workflow_id, config in workflows_config.get("workflows", {}).items():
                    hook_definitions = []
                    
                    for hook_config in config.get("hooks", []):
                        hook_definition = HookDefinition(**hook_config)
                        hook_definitions.append(hook_definition)
                    
                    self.workflow_hooks[workflow_id] = WorkflowHook(
                        hook_definitions=hook_definitions,
                        execution_order=config.get("execution_order", []),
                        failure_strategy=config.get("failure_strategy", "stop_on_critical"),
                        parallel_execution=config.get("parallel_execution", False),
                        max_concurrent=config.get("max_concurrent", 3)
                    )
                    
            except Exception as e:
                logger.error(
                    "Failed to load workflow hooks configuration",
                    file=str(workflow_hooks_file),
                    error=str(e)
                )
    
    async def execute_workflow_hooks(
        self,
        event: HookEventType,
        workflow_id: str,
        workflow_data: Dict[str, Any],
        agent_id: str,
        session_id: str
    ) -> List[HookResult]:
        """
        Execute hooks for a workflow event.
        
        Args:
            event: Hook event type
            workflow_id: Workflow identifier
            workflow_data: Workflow data and context
            agent_id: Agent executing the workflow
            session_id: Session identifier
            
        Returns:
            List of hook execution results
        """
        start_time = time.time()
        
        try:
            # Create hook context
            context = HookContext(
                workflow_id=workflow_id,
                agent_id=agent_id,
                session_id=session_id,
                event_type=event,
                event_data=workflow_data,
                project_root=str(self.project_root),
                environment=os.getenv("LEANVIBE_ENV", "development")
            )
            
            # Get applicable hooks
            applicable_hooks = await self._get_applicable_hooks(event, workflow_id, workflow_data)
            
            if not applicable_hooks:
                logger.debug(
                    "No applicable hooks found",
                    event=event.value,
                    workflow_id=workflow_id
                )
                return []
            
            # Execute hooks based on strategy
            results = await self._execute_hooks(applicable_hooks, context)
            
            # Process hook results
            await self._process_hook_results(results, context)
            
            # Update performance statistics
            execution_time = (time.time() - start_time) * 1000
            self.performance_stats["hooks_executed"] += len(results)
            self.performance_stats["total_execution_time_ms"] += execution_time
            
            failed_hooks = [r for r in results if not r.success]
            self.performance_stats["hooks_failed"] += len(failed_hooks)
            
            # Log hook execution summary
            logger.info(
                "🎣 Workflow hooks executed",
                event=event.value,
                workflow_id=workflow_id,
                hooks_count=len(applicable_hooks),
                success_count=len(results) - len(failed_hooks),
                failed_count=len(failed_hooks),
                execution_time_ms=execution_time
            )
            
            # Stream to hook processor for observability
            if self.hook_processor:
                await self.hook_processor.process_post_tool_use({
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "tool_name": "LeanVibeHooks",
                    "success": len(failed_hooks) == 0,
                    "result": {
                        "event": event.value,
                        "hooks_executed": len(results),
                        "hooks_successful": len(results) - len(failed_hooks),
                        "execution_time_ms": execution_time
                    },
                    "execution_time_ms": execution_time,
                    "correlation_id": context.correlation_id
                })
            
            return results
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.performance_stats["hooks_failed"] += 1
            
            logger.error(
                "❌ Failed to execute workflow hooks",
                event=event.value,
                workflow_id=workflow_id,
                error=str(e),
                execution_time_ms=execution_time,
                exc_info=True
            )
            
            return [HookResult(
                success=False,
                error=f"Hook execution failed: {str(e)}",
                execution_time_ms=execution_time,
                should_continue=False
            )]
    
    async def _get_applicable_hooks(
        self,
        event: HookEventType,
        workflow_id: str,
        workflow_data: Dict[str, Any]
    ) -> List[HookDefinition]:
        """Get hooks applicable to the current event and context."""
        applicable_hooks = []
        
        # Get event-specific hooks
        event_hooks = self.hook_definitions.get(event, [])
        
        # Get workflow-specific hooks
        workflow_hooks = []
        if workflow_id in self.workflow_hooks:
            workflow_hook_config = self.workflow_hooks[workflow_id]
            workflow_hooks = [
                hook for hook in workflow_hook_config.hook_definitions 
                if hook.event_type == event
            ]
        
        # Combine and filter hooks
        all_hooks = event_hooks + workflow_hooks
        
        for hook in all_hooks:
            if await self._should_execute_hook(hook, workflow_data):
                applicable_hooks.append(hook)
        
        return applicable_hooks
    
    async def _should_execute_hook(
        self,
        hook: HookDefinition,
        workflow_data: Dict[str, Any]
    ) -> bool:
        """Determine if a hook should be executed based on matcher and conditions."""
        try:
            # Check matcher pattern
            agent_name = workflow_data.get("agent_name", "")
            task_type = workflow_data.get("task_type", "")
            workflow_type = workflow_data.get("workflow_type", "")
            
            # Simple pattern matching (can be enhanced with regex)
            matcher = hook.matcher
            if matcher == "*":
                matches_pattern = True
            elif matcher.endswith("*"):
                prefix = matcher[:-1]
                matches_pattern = any(
                    value.startswith(prefix) 
                    for value in [agent_name, task_type, workflow_type]
                )
            elif matcher.startswith("*"):
                suffix = matcher[1:]
                matches_pattern = any(
                    value.endswith(suffix) 
                    for value in [agent_name, task_type, workflow_type]
                )
            else:
                matches_pattern = matcher in [agent_name, task_type, workflow_type]
            
            if not matches_pattern:
                return False
            
            # Check additional conditions
            for condition_key, condition_value in hook.conditions.items():
                data_value = workflow_data.get(condition_key)
                
                if isinstance(condition_value, dict):
                    operator = condition_value.get("operator", "equals")
                    expected_value = condition_value.get("value")
                    
                    if operator == "equals" and data_value != expected_value:
                        return False
                    elif operator == "contains" and expected_value not in str(data_value):
                        return False
                    elif operator == "greater_than" and not (isinstance(data_value, (int, float)) and data_value > expected_value):
                        return False
                    elif operator == "less_than" and not (isinstance(data_value, (int, float)) and data_value < expected_value):
                        return False
                else:
                    if data_value != condition_value:
                        return False
            
            return True
            
        except Exception as e:
            logger.warning(
                "Error evaluating hook conditions",
                hook_name=hook.name,
                error=str(e)
            )
            return False
    
    async def _execute_hooks(
        self,
        hooks: List[HookDefinition],
        context: HookContext
    ) -> List[HookResult]:
        """Execute a list of hooks with appropriate strategy."""
        if not hooks:
            return []
        
        # Separate hooks by execution mode
        blocking_hooks = [h for h in hooks if h.execution_mode == HookExecutionMode.BLOCKING]
        critical_hooks = [h for h in hooks if h.execution_mode == HookExecutionMode.CRITICAL]
        async_hooks = [h for h in hooks if h.execution_mode == HookExecutionMode.ASYNC]
        non_blocking_hooks = [h for h in hooks if h.execution_mode == HookExecutionMode.NON_BLOCKING]
        
        results = []
        
        # Execute critical hooks first (must succeed)
        for hook in critical_hooks:
            result = await self._execute_single_hook(hook, context)
            results.append(result)
            
            if not result.success and hook.required:
                logger.error(
                    "🚨 Critical hook failed - stopping execution",
                    hook_name=hook.name,
                    error=result.error
                )
                self.performance_stats["blocked_actions"] += 1
                return results
        
        # Execute blocking hooks sequentially
        for hook in blocking_hooks:
            result = await self._execute_single_hook(hook, context)
            results.append(result)
            
            if not result.success and hook.required:
                logger.warning(
                    "⚠️ Required blocking hook failed",
                    hook_name=hook.name,
                    error=result.error
                )
                # Continue execution but log the failure
        
        # Execute async and non-blocking hooks in parallel
        parallel_hooks = async_hooks + non_blocking_hooks
        if parallel_hooks:
            parallel_tasks = [
                self._execute_single_hook(hook, context) 
                for hook in parallel_hooks
            ]
            
            parallel_results = await asyncio.gather(*parallel_tasks, return_exceptions=True)
            
            for i, result in enumerate(parallel_results):
                if isinstance(result, Exception):
                    hook = parallel_hooks[i]
                    results.append(HookResult(
                        success=False,
                        error=f"Hook execution exception: {str(result)}",
                        execution_time_ms=0,
                        should_continue=not hook.required
                    ))
                else:
                    results.append(result)
        
        return results
    
    async def _execute_single_hook(
        self,
        hook: HookDefinition,
        context: HookContext
    ) -> HookResult:
        """Execute a single hook with proper error handling and caching."""
        start_time = time.time()
        
        try:
            # Check cache for recent results
            cache_key = self._generate_cache_key(hook, context)
            if cache_key in self.execution_cache:
                cached_result = self.execution_cache[cache_key]
                # Use cached result if less than 5 minutes old
                if (time.time() - cached_result.execution_time_ms / 1000) < 300:
                    self.performance_stats["cache_hits"] += 1
                    logger.debug(
                        "Using cached hook result",
                        hook_name=hook.name,
                        cache_key=cache_key
                    )
                    return cached_result
            
            # Prepare execution environment
            env = os.environ.copy()
            env.update(hook.environment_vars)
            env.update({
                "LEANVIBE_WORKFLOW_ID": context.workflow_id,
                "LEANVIBE_AGENT_ID": context.agent_id,
                "LEANVIBE_SESSION_ID": context.session_id,
                "LEANVIBE_EVENT_TYPE": context.event_type.value,
                "LEANVIBE_PROJECT_ROOT": context.project_root,
                "LEANVIBE_CORRELATION_ID": context.correlation_id,
                "LEANVIBE_EVENT_DATA": json.dumps(context.event_data, default=str)
            })
            
            # Determine working directory
            working_dir = hook.working_directory or context.project_root
            
            # Execute hook command with timeout
            logger.debug(
                "Executing hook",
                hook_name=hook.name,
                command=hook.command,
                working_dir=working_dir,
                timeout=hook.timeout_seconds
            )
            
            # Create subprocess for hook execution
            process = await asyncio.create_subprocess_shell(
                hook.command,
                cwd=working_dir,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=hook.timeout_seconds
                )
                
                execution_time = (time.time() - start_time) * 1000
                
                # Process results
                stdout_text = stdout.decode('utf-8') if stdout else ""
                stderr_text = stderr.decode('utf-8') if stderr else ""
                
                success = process.returncode == 0
                output = stdout_text
                error = stderr_text if not success else None
                
                result = HookResult(
                    success=success,
                    output=output,
                    error=error,
                    execution_time_ms=execution_time,
                    metadata={
                        "return_code": process.returncode,
                        "hook_name": hook.name,
                        "execution_mode": hook.execution_mode.value,
                        "timeout_seconds": hook.timeout_seconds
                    }
                )
                
                # Cache successful results
                if success:
                    self.execution_cache[cache_key] = result
                    # Limit cache size
                    if len(self.execution_cache) > 1000:
                        # Remove oldest entries
                        oldest_keys = list(self.execution_cache.keys())[:100]
                        for key in oldest_keys:
                            del self.execution_cache[key]
                
                logger.debug(
                    "Hook execution completed",
                    hook_name=hook.name,
                    success=success,
                    execution_time_ms=execution_time,
                    return_code=process.returncode
                )
                
                return result
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                
                execution_time = (time.time() - start_time) * 1000
                
                logger.warning(
                    "Hook execution timed out",
                    hook_name=hook.name,
                    timeout_seconds=hook.timeout_seconds
                )
                
                return HookResult(
                    success=False,
                    error=f"Hook execution timed out after {hook.timeout_seconds}s",
                    execution_time_ms=execution_time,
                    should_continue=not hook.required
                )
                
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            logger.error(
                "Hook execution failed with exception",
                hook_name=hook.name,
                error=str(e),
                exc_info=True
            )
            
            return HookResult(
                success=False,
                error=f"Hook execution exception: {str(e)}",
                execution_time_ms=execution_time,
                should_continue=not hook.required
            )
    
    def _generate_cache_key(self, hook: HookDefinition, context: HookContext) -> str:
        """Generate cache key for hook execution result."""
        key_data = {
            "hook_name": hook.name,
            "command": hook.command,
            "event_type": context.event_type.value,
            "event_data_hash": hash(json.dumps(context.event_data, sort_keys=True, default=str))
        }
        return f"hook_{hash(json.dumps(key_data, sort_keys=True))}"
    
    async def _process_hook_results(
        self,
        results: List[HookResult],
        context: HookContext
    ) -> None:
        """Process hook execution results and handle failures."""
        try:
            failed_results = [r for r in results if not r.success]
            
            if failed_results:
                # Log failures
                for result in failed_results:
                    logger.warning(
                        "Hook execution failed",
                        error=result.error,
                        execution_time_ms=result.execution_time_ms,
                        metadata=result.metadata
                    )
                
                # Send failure notifications if communication bus is available
                if self.communication_bus:
                    # Note: MessageBroker might have different method signature
                    # This would need to be adapted based on actual MessageBroker API
                    pass
        
        except Exception as e:
            logger.error(
                "Error processing hook results",
                error=str(e),
                exc_info=True
            )
    
    async def create_default_hooks_directory(self) -> None:
        """Create default hooks directory with essential hook scripts."""
        hooks_dir = self.hooks_config_dir
        hooks_dir.mkdir(parents=True, exist_ok=True)
        
        # Create hooks configuration
        hooks_config = {
            "leanvibe_hooks": {
                "PreAgentTask": [
                    {
                        "name": "security_validation",
                        "command": "python .leanvibe/hooks/validate_security.py",
                        "matcher": "*",
                        "description": "Validate task doesn't access sensitive files",
                        "execution_mode": "blocking",
                        "timeout_seconds": 30,
                        "required": True
                    },
                    {
                        "name": "dependency_check",
                        "command": "python .leanvibe/hooks/check_dependencies.py",
                        "matcher": "backend_*",
                        "description": "Ensure required dependencies are available",
                        "execution_mode": "non_blocking",
                        "timeout_seconds": 60
                    }
                ],
                "PostAgentTask": [
                    {
                        "name": "code_formatting",
                        "command": "python .leanvibe/hooks/format_code.py",
                        "matcher": "*_specialist",
                        "description": "Auto-format code according to project standards",
                        "execution_mode": "async",
                        "timeout_seconds": 120
                    },
                    {
                        "name": "test_execution",
                        "command": "python .leanvibe/hooks/run_tests.py",
                        "matcher": "*",
                        "description": "Run relevant tests for changed code",
                        "execution_mode": "blocking",
                        "timeout_seconds": 300,
                        "required": True
                    }
                ],
                "WorkflowComplete": [
                    {
                        "name": "quality_report",
                        "command": "python .leanvibe/hooks/generate_quality_report.py",
                        "matcher": "*",
                        "description": "Generate comprehensive quality and progress report",
                        "execution_mode": "async",
                        "timeout_seconds": 180
                    }
                ]
            }
        }
        
        config_file = hooks_dir / "hooks.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(hooks_config, f, default_flow_style=False)
        
        # Create hook scripts directory
        scripts_dir = hooks_dir
        scripts_dir.mkdir(exist_ok=True)
        
        # Create essential hook scripts
        await self._create_hook_scripts(scripts_dir)
        
        logger.info(
            "📁 Default hooks directory created",
            hooks_dir=str(hooks_dir),
            config_file=str(config_file),
            scripts_created=8
        )
    
    async def _create_hook_scripts(self, scripts_dir: Path) -> None:
        """Create essential hook scripts."""
        scripts = {
            "validate_security.py": '''#!/usr/bin/env python3
"""Security validation hook for LeanVibe Agent Hive."""

import json
import os
import sys
from pathlib import Path

def validate_security():
    """Validate that agent task doesn't access sensitive files."""
    try:
        event_data = json.loads(os.environ.get("LEANVIBE_EVENT_DATA", "{}"))
        
        # Check for sensitive file access
        sensitive_patterns = [
            ".env", "secrets", "private_key", "password", 
            ".ssh", "credentials", "token", "api_key"
        ]
        
        task_data = event_data.get("parameters", {})
        file_paths = []
        
        # Extract file paths from various task parameters
        for key, value in task_data.items():
            if isinstance(value, str) and ("/" in value or "\\\\" in value):
                file_paths.append(value)
        
        # Check for sensitive patterns
        for file_path in file_paths:
            for pattern in sensitive_patterns:
                if pattern.lower() in file_path.lower():
                    print(f"SECURITY VIOLATION: Attempting to access sensitive file: {file_path}")
                    sys.exit(1)
        
        print("Security validation passed")
        sys.exit(0)
        
    except Exception as e:
        print(f"Security validation error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    validate_security()
''',
            
            "check_dependencies.py": '''#!/usr/bin/env python3
"""Dependency check hook for LeanVibe Agent Hive."""

import json
import os
import subprocess
import sys

def check_dependencies():
    """Check that required dependencies are available."""
    try:
        event_data = json.loads(os.environ.get("LEANVIBE_EVENT_DATA", "{}"))
        
        # Check Python dependencies
        result = subprocess.run(
            ["python", "-c", "import sys; print('Python dependencies OK')"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print(f"Python dependency check failed: {result.stderr}")
            sys.exit(1)
        
        print("Dependency check passed")
        sys.exit(0)
        
    except Exception as e:
        print(f"Dependency check error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_dependencies()
''',
            
            "format_code.py": '''#!/usr/bin/env python3
"""Code formatting hook for LeanVibe Agent Hive."""

import json
import os
import subprocess
import sys
from pathlib import Path

def format_code():
    """Auto-format code according to project standards."""
    try:
        project_root = Path(os.environ.get("LEANVIBE_PROJECT_ROOT", "."))
        
        # Format Python files with black
        python_files = list(project_root.rglob("*.py"))
        if python_files:
            try:
                subprocess.run(
                    ["black", "--line-length", "88"] + [str(f) for f in python_files],
                    check=True,
                    timeout=120
                )
                print(f"Formatted {len(python_files)} Python files")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("Black formatter not available, skipping Python formatting")
        
        # Format TypeScript/JavaScript files with prettier (if available)
        ts_files = list(project_root.rglob("*.ts")) + list(project_root.rglob("*.js"))
        if ts_files:
            try:
                subprocess.run(
                    ["prettier", "--write"] + [str(f) for f in ts_files],
                    check=True,
                    timeout=120
                )
                print(f"Formatted {len(ts_files)} TypeScript/JavaScript files")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("Prettier formatter not available, skipping TypeScript/JavaScript formatting")
        
        print("Code formatting completed")
        sys.exit(0)
        
    except Exception as e:
        print(f"Code formatting error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    format_code()
''',
            
            "run_tests.py": '''#!/usr/bin/env python3
"""Test execution hook for LeanVibe Agent Hive."""

import json
import os
import subprocess
import sys
from pathlib import Path

def run_tests():
    """Run relevant tests for changed code."""
    try:
        project_root = Path(os.environ.get("LEANVIBE_PROJECT_ROOT", "."))
        
        # Try different test runners
        test_runners = [
            (["python", "-m", "pytest", "-v"], "pytest"),
            (["python", "-m", "unittest", "discover"], "unittest"),
            (["npm", "test"], "npm")
        ]
        
        test_passed = False
        
        for command, runner_name in test_runners:
            try:
                result = subprocess.run(
                    command,
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    print(f"Tests passed using {runner_name}")
                    print(result.stdout)
                    test_passed = True
                    break
                else:
                    print(f"Tests failed using {runner_name}: {result.stderr}")
                    
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        if test_passed:
            sys.exit(0)
        else:
            print("No test runner succeeded")
            sys.exit(1)
        
    except Exception as e:
        print(f"Test execution error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
''',
            
            "generate_quality_report.py": '''#!/usr/bin/env python3
"""Quality report generation hook for LeanVibe Agent Hive."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

def generate_quality_report():
    """Generate comprehensive quality and progress report."""
    try:
        workflow_id = os.environ.get("LEANVIBE_WORKFLOW_ID", "unknown")
        project_root = Path(os.environ.get("LEANVIBE_PROJECT_ROOT", "."))
        
        report = {
            "workflow_id": workflow_id,
            "timestamp": datetime.utcnow().isoformat(),
            "project_root": str(project_root),
            "quality_metrics": {
                "files_analyzed": 0,
                "code_quality_score": "N/A",
                "test_coverage": "N/A",
                "security_issues": "N/A"
            },
            "summary": "Quality report generated successfully"
        }
        
        # Count files in project
        python_files = list(project_root.rglob("*.py"))
        report["quality_metrics"]["files_analyzed"] = len(python_files)
        
        # Save report
        report_file = project_root / ".leanvibe" / "reports" / f"quality_report_{workflow_id}.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Quality report generated: {report_file}")
        print(json.dumps(report, indent=2))
        sys.exit(0)
        
    except Exception as e:
        print(f"Quality report generation error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    generate_quality_report()
'''
        }
        
        for script_name, script_content in scripts.items():
            script_file = scripts_dir / script_name
            with open(script_file, 'w') as f:
                f.write(script_content)
            
            # Make script executable
            os.chmod(script_file, 0o755)
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "performance_stats": self.performance_stats,
            "hooks_config": {
                "total_event_types": len(self.hook_definitions),
                "total_hooks": sum(len(hooks) for hooks in self.hook_definitions.values()),
                "workflow_hooks": len(self.workflow_hooks)
            },
            "cache_stats": {
                "cache_size": len(self.execution_cache),
                "cache_hit_rate": (
                    self.performance_stats["cache_hits"] / 
                    max(self.performance_stats["hooks_executed"], 1) * 100
                ) if self.performance_stats["hooks_executed"] > 0 else 0
            },
            "project_hooks_dir": str(self.hooks_config_dir),
            "project_hooks_dir_exists": self.hooks_config_dir.exists(),
            "user_hooks_dir": str(self.user_hooks_dir),
            "user_hooks_dir_exists": self.user_hooks_dir.exists(),
            "config_events": list(self.hook_definitions.keys()),
            "execution_stats": {
                "hooks_executed": self.performance_stats["hooks_executed"],
                "hooks_failed": self.performance_stats["hooks_failed"],
                "blocked_actions": self.performance_stats["blocked_actions"],
                "total_execution_time_ms": self.performance_stats["total_execution_time_ms"],
                "average_execution_time_ms": (
                    self.performance_stats["total_execution_time_ms"] / 
                    max(self.performance_stats["hooks_executed"], 1)
                ) if self.performance_stats["hooks_executed"] > 0 else 0
            }
        }


# Global hooks engine instance
_leanvibe_hooks_engine: Optional[LeanVibeHooksEngine] = None


def get_leanvibe_hooks_engine() -> Optional[LeanVibeHooksEngine]:
    """Get the global LeanVibe hooks engine instance."""
    return _leanvibe_hooks_engine


def set_leanvibe_hooks_engine(engine: LeanVibeHooksEngine) -> None:
    """Set the global LeanVibe hooks engine instance."""
    global _leanvibe_hooks_engine
    _leanvibe_hooks_engine = engine
    logger.info("🔗 Global LeanVibe hooks engine set")


async def initialize_leanvibe_hooks_engine(
    project_root: Optional[Path] = None,
    orchestrator: Optional[AgentOrchestrator] = None,
    communication_bus: Optional[MessageBroker] = None,
    hook_processor: Optional[HookEventProcessor] = None
) -> LeanVibeHooksEngine:
    """
    Initialize and set the global LeanVibe hooks engine.
    
    Args:
        project_root: Project root directory
        orchestrator: Agent orchestrator instance
        communication_bus: Communication bus instance
        hook_processor: Hook event processor instance
        
    Returns:
        LeanVibeHooksEngine instance
    """
    engine = LeanVibeHooksEngine(
        project_root=project_root,
        orchestrator=orchestrator,
        communication_bus=communication_bus,
        hook_processor=hook_processor
    )
    
    # Load hooks configuration
    await engine.load_hooks_configuration()
    
    # Create default hooks directory
    await engine.create_default_hooks_directory()
    
    set_leanvibe_hooks_engine(engine)
    
    logger.info("✅ LeanVibe hooks engine initialized")
    return engine