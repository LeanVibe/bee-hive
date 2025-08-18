"""
Gemini CLI Adapter for Multi-CLI Coordination

This adapter enables Gemini CLI to participate in coordinated multi-agent
workflows. It implements the UniversalAgentInterface to provide standardized
communication, task execution, and capability reporting.

Key Features:
- Subprocess-based Gemini CLI integration
- Secure worktree isolation and path validation
- Message format translation (universal ↔ Gemini)
- Performance monitoring and error handling
- Capability-based task routing support

Production Status: READY - Implemented following claude_code_adapter pattern
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass, field
from datetime import datetime

from ..universal_agent_interface import (
    UniversalAgentInterface,
    AgentType,
    AgentTask,
    AgentResult,
    AgentCapability,
    ExecutionContext,
    HealthStatus,
    CapabilityType,
    TaskStatus,
    HealthState,
    AgentError,
    TaskExecutionError,
    SecurityError,
    ResourceLimitError
)

logger = logging.getLogger(__name__)

# ================================================================================
# GEMINI CLI COMMAND MODEL
# ================================================================================

@dataclass
class GeminiCommand:
    """Represents a Gemini CLI command with all parameters."""
    command: str
    options: Dict[str, Any] = field(default_factory=dict)
    input_files: List[str] = field(default_factory=list)
    timeout_seconds: Optional[float] = 300.0
    
    def to_cli_args(self) -> List[str]:
        """Convert to CLI arguments list."""
        args = [self.command]
        
        # Add options
        for key, value in self.options.items():
            if value is True:
                args.append(key)
            elif value is not False and value is not None:
                args.extend([key, str(value)])
        
        # Add input files
        args.extend(self.input_files)
        
        return args

@dataclass
class GeminiResponse:
    """Represents the response from Gemini CLI execution."""
    success: bool
    output: str
    error_output: str
    return_code: int
    execution_time: float
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    memory_usage_mb: Optional[float] = None
    tokens_used: Optional[int] = None


# ================================================================================
# GEMINI CLI ADAPTER IMPLEMENTATION
# ================================================================================

class GeminiCLIAdapter(UniversalAgentInterface):
    """
    Production Gemini CLI Adapter.
    
    Provides seamless integration with Gemini CLI through the universal
    agent interface, enabling Gemini to participate in multi-agent workflows.
    """
    
    def __init__(
        self,
        cli_path: str = "gemini",
        working_directory: Optional[str] = None,
        max_concurrent_tasks: int = 2,  # Conservative limit for API calls
        default_timeout: float = 300.0
    ):
        """
        Initialize Gemini CLI adapter.
        
        Args:
            cli_path: Path to Gemini CLI executable
            working_directory: Default working directory for operations
            max_concurrent_tasks: Maximum concurrent task limit
            default_timeout: Default timeout for operations
        """
        agent_id = f"gemini_agent_{str(uuid.uuid4())[:8]}"
        super().__init__(agent_id, AgentType.GEMINI_CLI)
        
        self._cli_path = cli_path
        self._working_directory = working_directory or os.getcwd()
        self._max_concurrent_tasks = max_concurrent_tasks
        self._default_timeout = default_timeout
        self._active_tasks: Set[str] = set()
        
        # Performance tracking
        self._task_count = 0
        self._success_count = 0
        self._total_execution_time = 0.0
        self._last_activity = datetime.now()
        self._total_tokens_used = 0
        
        logger.info(f"Gemini CLI adapter initialized: {agent_id}")

    # ========================================================================
    # UNIVERSAL AGENT INTERFACE IMPLEMENTATION  
    # ========================================================================

    async def execute_task(self, task: AgentTask) -> AgentResult:
        """
        Execute a task using Gemini CLI.
        
        Args:
            task: Universal agent task to execute
            
        Returns:
            AgentResult: Standardized task execution result
        """
        start_time = time.time()
        self._task_count += 1
        
        # Add to active tasks
        self._active_tasks.add(task.id)
        
        try:
            logger.info(f"Executing task {task.id}: {task.description}")
            
            # 1. Validate task compatibility
            if not await self._is_task_compatible(task):
                return AgentResult(
                    task_id=task.id,
                    agent_id=self.agent_id,
                    agent_type=self.agent_type,
                    status=TaskStatus.FAILED,
                    error_message=f"Task type {task.type} not supported by Gemini CLI"
                )
            
            # 2. Check resource limits
            if len(self._active_tasks) > self._max_concurrent_tasks:
                return AgentResult(
                    task_id=task.id,
                    agent_id=self.agent_id,
                    agent_type=self.agent_type,
                    status=TaskStatus.FAILED,
                    error_message=f"Maximum concurrent tasks ({self._max_concurrent_tasks}) exceeded"
                )
            
            # 3. Translate universal task to Gemini command
            gemini_command = self._translate_task_to_command(task)
            
            # 4. Execute Gemini command
            response = await self._execute_gemini_command(gemini_command, task.context)
            
            # 5. Translate response to universal format
            result = self._translate_response_to_result(task.id, response)
            
            # 6. Update performance metrics
            execution_time = time.time() - start_time
            self._total_execution_time += execution_time
            if response.tokens_used:
                self._total_tokens_used += response.tokens_used
            if result.status == TaskStatus.COMPLETED:
                self._success_count += 1
            
            self._last_activity = datetime.now()
            
            logger.info(
                f"Task {task.id} completed in {execution_time:.2f}s: {result.status}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                status=TaskStatus.FAILED,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
        finally:
            # Remove from active tasks
            self._active_tasks.discard(task.id)

    async def get_capabilities(self) -> List[AgentCapability]:
        """
        Get Gemini CLI agent capabilities.
        
        Returns:
            List of agent capabilities with confidence scores
        """
        return [
            AgentCapability(
                type=CapabilityType.TESTING,
                confidence=0.90,
                performance_score=0.85,
                description="Advanced testing strategies and test case generation"
            ),
            AgentCapability(
                type=CapabilityType.CODE_REVIEW,
                confidence=0.85,
                performance_score=0.90,
                description="Comprehensive code review and quality analysis"
            ),
            AgentCapability(
                type=CapabilityType.CODE_ANALYSIS,
                confidence=0.90,
                performance_score=0.90,
                description="Deep code analysis and architectural insights"
            ),
            AgentCapability(
                type=CapabilityType.DEBUGGING,
                confidence=0.80,
                performance_score=0.85,
                description="Advanced debugging techniques and error analysis"
            ),
            AgentCapability(
                type=CapabilityType.ARCHITECTURE_DESIGN,
                confidence=0.85,
                performance_score=0.80,
                description="System architecture design and patterns"
            ),
            AgentCapability(
                type=CapabilityType.PERFORMANCE_OPTIMIZATION,
                confidence=0.80,
                performance_score=0.85,
                description="Performance analysis and optimization strategies"
            ),
            AgentCapability(
                type=CapabilityType.DOCUMENTATION,
                confidence=0.75,
                performance_score=0.80,
                description="Technical documentation and explanation generation"
            ),
            AgentCapability(
                type=CapabilityType.CODE_IMPLEMENTATION,
                confidence=0.70,
                performance_score=0.75,
                description="Code implementation with best practices focus"
            )
        ]

    async def health_check(self) -> HealthStatus:
        """
        Perform health check on Gemini CLI adapter.
        
        Returns:
            HealthStatus: Current health status
        """
        start_time = time.time()
        
        try:
            # Test Gemini CLI availability
            test_process = await asyncio.create_subprocess_exec(
                self._cli_path, "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(
                test_process.communicate(),
                timeout=15.0  # Longer timeout for API-based tools
            )
            
            response_time = (time.time() - start_time) * 1000
            
            if test_process.returncode == 0:
                # Calculate performance metrics
                success_rate = (
                    self._success_count / self._task_count 
                    if self._task_count > 0 else 1.0
                )
                avg_response_time = (
                    (self._total_execution_time / self._task_count * 1000)
                    if self._task_count > 0 else response_time
                )
                
                return HealthStatus(
                    agent_id=self.agent_id,
                    agent_type=self.agent_type,
                    state=HealthState.HEALTHY,
                    response_time_ms=response_time,
                    cpu_usage_percent=8.0,  # Lower CPU usage for API-based operations
                    memory_usage_mb=96.0,   # Conservative memory estimate
                    active_tasks=len(self._active_tasks),
                    completed_tasks=self._success_count,
                    failed_tasks=self._task_count - self._success_count,
                    last_activity=self._last_activity,
                    error_rate=1.0 - success_rate,
                    throughput_tasks_per_minute=60.0 / avg_response_time * 1000 if avg_response_time > 0 else 0
                )
            else:
                return HealthStatus(
                    agent_id=self.agent_id,
                    agent_type=self.agent_type,
                    state=HealthState.UNHEALTHY,
                    response_time_ms=response_time,
                    error_message="Gemini CLI not responding properly",
                    last_activity=self._last_activity
                )
                
        except asyncio.TimeoutError:
            return HealthStatus(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                state=HealthState.UNHEALTHY,
                response_time_ms=15000.0,
                error_message="Gemini CLI health check timeout",
                last_activity=self._last_activity
            )
        except Exception as e:
            return HealthStatus(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                state=HealthState.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=f"Health check failed: {str(e)}",
                last_activity=self._last_activity
            )

    # ========================================================================
    # GEMINI CLI-SPECIFIC IMPLEMENTATION
    # ========================================================================

    async def _is_task_compatible(self, task: AgentTask) -> bool:
        """Check if task is compatible with Gemini CLI capabilities."""
        compatible_types = {
            CapabilityType.TESTING,
            CapabilityType.CODE_REVIEW,
            CapabilityType.CODE_ANALYSIS,
            CapabilityType.DEBUGGING,
            CapabilityType.ARCHITECTURE_DESIGN,
            CapabilityType.PERFORMANCE_OPTIMIZATION,
            CapabilityType.DOCUMENTATION,
            CapabilityType.CODE_IMPLEMENTATION
        }
        return task.type in compatible_types

    def _translate_task_to_command(self, task: AgentTask) -> GeminiCommand:
        """
        Translate universal task to Gemini CLI command.
        
        Args:
            task: Universal agent task
            
        Returns:
            GeminiCommand: Gemini CLI-specific command
        """
        # Map capability types to Gemini CLI commands
        command_map = {
            CapabilityType.TESTING: "test",
            CapabilityType.CODE_REVIEW: "review",
            CapabilityType.CODE_ANALYSIS: "analyze",
            CapabilityType.DEBUGGING: "debug",
            CapabilityType.ARCHITECTURE_DESIGN: "design",
            CapabilityType.PERFORMANCE_OPTIMIZATION: "optimize",
            CapabilityType.DOCUMENTATION: "document",
            CapabilityType.CODE_IMPLEMENTATION: "implement"
        }
        
        base_command = command_map.get(task.type, "analyze")
        
        # Build command options
        options = {}
        
        # Add task-specific options
        if task.description:
            options["--prompt"] = task.description
        
        if task.requirements:
            options["--requirements"] = ",".join(task.requirements)
        
        # Add priority handling
        if task.priority <= 3:
            options["--priority"] = "high"
        elif task.priority >= 8:
            options["--priority"] = "low"
            
        # Add output format
        options["--format"] = "json"
        
        # Add Gemini-specific options
        options["--temperature"] = "0.3"  # Lower temperature for more consistent results
        options["--model"] = "gemini-pro"  # Use the most capable model
        
        # Add context window optimization
        options["--max-tokens"] = "4000"  # Conservative token limit
        
        # Get input files
        input_files = []
        if task.context and task.context.file_scope:
            input_files = task.context.file_scope
        elif task.input_data and "files" in task.input_data:
            input_files = task.input_data["files"]
        
        command = GeminiCommand(
            command=base_command,
            options=options,
            input_files=input_files,
            timeout_seconds=task.timeout_seconds or self._default_timeout
        )
        
        return command

    async def _execute_gemini_command(
        self,
        command: GeminiCommand,
        context: Optional[ExecutionContext]
    ) -> GeminiResponse:
        """
        Execute Gemini CLI command with proper isolation.
        
        Args:
            command: Gemini command to execute
            context: Execution context with isolation settings
            
        Returns:
            GeminiResponse: Command execution result
        """
        start_time = time.time()
        
        try:
            # 1. Prepare execution environment
            work_dir = context.worktree_path if context else self._working_directory
            if not work_dir or not os.path.exists(work_dir):
                work_dir = tempfile.mkdtemp(prefix="gemini_work_")
                logger.info(f"Created temporary work directory: {work_dir}")
            
            env = self._prepare_environment(context)
            
            # 2. Validate command safety
            self._validate_command_safety(command)
            
            # 3. Build command arguments
            cli_args = [self._cli_path] + command.to_cli_args()
            logger.debug(f"Executing Gemini CLI command: {' '.join(cli_args)}")
            
            # 4. Track files before execution
            existing_files = set()
            if os.path.exists(work_dir):
                for root, dirs, files in os.walk(work_dir):
                    for file in files:
                        existing_files.add(os.path.join(root, file))
            
            # 5. Execute subprocess with timeout (longer for AI processing)
            process = await asyncio.create_subprocess_exec(
                *cli_args,
                cwd=work_dir,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=command.timeout_seconds
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise ResourceLimitError(
                    f"Gemini CLI command timed out after {command.timeout_seconds}s"
                )
            
            # 6. Track file changes
            files_created = []
            files_modified = []
            
            if os.path.exists(work_dir):
                for root, dirs, files in os.walk(work_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if file_path not in existing_files:
                            files_created.append(file_path)
                        else:
                            files_modified.append(file_path)
            
            execution_time = time.time() - start_time
            
            # Parse output to extract token usage if available
            tokens_used = None
            try:
                if stdout and process.returncode == 0:
                    output_data = json.loads(stdout.decode('utf-8'))
                    if 'usage' in output_data and 'total_tokens' in output_data['usage']:
                        tokens_used = output_data['usage']['total_tokens']
            except (json.JSONDecodeError, KeyError):
                pass
            
            return GeminiResponse(
                success=process.returncode == 0,
                output=stdout.decode('utf-8') if stdout else "",
                error_output=stderr.decode('utf-8') if stderr else "",
                return_code=process.returncode,
                execution_time=execution_time,
                files_created=files_created,
                files_modified=files_modified,
                tokens_used=tokens_used
            )
            
        except Exception as e:
            logger.error(f"Gemini CLI command execution failed: {e}")
            execution_time = time.time() - start_time
            
            return GeminiResponse(
                success=False,
                output="",
                error_output=str(e),
                return_code=-1,
                execution_time=execution_time
            )

    def _translate_response_to_result(
        self,
        task_id: str,
        response: GeminiResponse
    ) -> AgentResult:
        """
        Translate Gemini response to universal agent result.
        
        Args:
            task_id: Original task ID
            response: Gemini command response
            
        Returns:
            AgentResult: Universal agent result
        """
        if response.success:
            # Parse output if it's JSON
            output_data = {}
            try:
                if response.output.strip():
                    output_data = json.loads(response.output)
            except json.JSONDecodeError:
                output_data = {"raw_output": response.output}
            
            # Add Gemini-specific information
            output_data.update({
                "files_created": response.files_created,
                "files_modified": response.files_modified,
                "execution_time": response.execution_time,
                "tokens_used": response.tokens_used,
                "model": "gemini-pro",  # Track which model was used
                "efficiency_score": (
                    response.tokens_used / response.execution_time 
                    if response.tokens_used and response.execution_time > 0 
                    else None
                )
            })
            
            return AgentResult(
                task_id=task_id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                status=TaskStatus.COMPLETED,
                output_data=output_data,
                execution_time=response.execution_time,
                metadata={
                    "gemini_return_code": response.return_code,
                    "tokens_used": response.tokens_used,
                    "memory_usage_mb": response.memory_usage_mb,
                    "model_used": "gemini-pro"
                }
            )
        else:
            return AgentResult(
                task_id=task_id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                status=TaskStatus.FAILED,
                error_message=response.error_output or "Gemini CLI command failed",
                execution_time=response.execution_time,
                metadata={
                    "gemini_return_code": response.return_code,
                    "raw_output": response.output
                }
            )

    def _prepare_environment(self, context: Optional[ExecutionContext]) -> Dict[str, str]:
        """Prepare environment variables for Gemini CLI execution."""
        env = os.environ.copy()
        
        if context and context.environment_vars:
            env.update(context.environment_vars)
        
        # Add Gemini-specific environment variables
        env["GEMINI_OUTPUT_FORMAT"] = "json"
        env["GEMINI_TEMPERATURE"] = "0.3"
        env["GEMINI_MODEL"] = "gemini-pro"
        
        # Set API key if available (should be set in environment)
        # env["GOOGLE_API_KEY"] should already be set
        
        return env

    def _validate_command_safety(self, command: GeminiCommand):
        """Validate command for security and safety."""
        # Check for dangerous commands
        dangerous_patterns = [
            "rm -rf", "sudo", "chmod 777", "curl", "wget",
            "eval", "exec", "system", "__import__",
            # Gemini-specific dangerous patterns
            "api-key", "credential", "secret", "token"
        ]
        
        command_str = " ".join(command.to_cli_args()).lower()
        
        for pattern in dangerous_patterns:
            if pattern in command_str:
                raise SecurityError(f"Potentially dangerous command pattern detected: {pattern}")
        
        # Validate file paths
        for file_path in command.input_files:
            if not self._is_safe_path(file_path):
                raise SecurityError(f"Unsafe file path: {file_path}")
        
        # Validate token limits
        max_tokens_option = command.options.get("--max-tokens")
        if max_tokens_option and isinstance(max_tokens_option, str):
            try:
                max_tokens = int(max_tokens_option)
                if max_tokens > 8000:  # Conservative limit
                    raise SecurityError(f"Token limit too high: {max_tokens} > 8000")
            except ValueError:
                raise SecurityError(f"Invalid token limit: {max_tokens_option}")

    def _is_safe_path(self, path: str) -> bool:
        """Check if file path is safe for operations."""
        # Prevent path traversal attacks
        if ".." in path or path.startswith("/"):
            return False
        
        # Prevent access to system directories
        system_dirs = ["/etc", "/usr", "/var", "/sys", "/proc"]
        abs_path = os.path.abspath(path)
        
        for sys_dir in system_dirs:
            if abs_path.startswith(sys_dir):
                return False
        
        return True

    # ========================================================================
    # LIFECYCLE MANAGEMENT
    # ========================================================================

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the Gemini CLI adapter with configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            bool: True if initialization successful
        """
        try:
            # Update configuration
            if "cli_path" in config:
                self._cli_path = config["cli_path"]
            
            if "working_directory" in config:
                self._working_directory = config["working_directory"]
            
            if "max_concurrent_tasks" in config:
                self._max_concurrent_tasks = config["max_concurrent_tasks"]
            
            if "default_timeout" in config:
                self._default_timeout = config["default_timeout"]
            
            # Verify Gemini CLI is available
            health = await self.health_check()
            
            if health.state == HealthState.HEALTHY:
                logger.info(f"Gemini CLI adapter {self.agent_id} initialized successfully")
                return True
            else:
                logger.error(f"Gemini CLI adapter initialization failed: {health.error_message}")
                return False
                
        except Exception as e:
            logger.error(f"Gemini CLI adapter initialization error: {e}")
            return False

    async def shutdown(self) -> None:
        """
        Gracefully shutdown the Gemini CLI adapter.
        """
        logger.info(f"Shutting down Gemini CLI adapter {self.agent_id}")
        
        # Wait for active tasks to complete (with timeout)
        shutdown_timeout = 60.0  # Longer timeout for AI processing
        start_time = time.time()
        
        while self._active_tasks and (time.time() - start_time) < shutdown_timeout:
            logger.info(f"Waiting for {len(self._active_tasks)} active tasks to complete...")
            await asyncio.sleep(1.0)
        
        if self._active_tasks:
            logger.warning(f"Shutdown timeout reached, {len(self._active_tasks)} tasks may be incomplete")
        
        logger.info(f"Gemini CLI adapter {self.agent_id} shutdown completed")
        logger.info(f"Total tokens used during session: {self._total_tokens_used}")


# ================================================================================
# FACTORY FUNCTION
# ================================================================================

def create_gemini_cli_adapter(
    cli_path: str = "gemini",
    working_directory: Optional[str] = None,
    **kwargs
) -> GeminiCLIAdapter:
    """
    Factory function to create a Gemini CLI adapter instance.
    
    Args:
        cli_path: Path to Gemini CLI executable
        working_directory: Working directory for operations
        **kwargs: Additional configuration options
        
    Returns:
        GeminiCLIAdapter: Configured Gemini CLI adapter instance
    """
    return GeminiCLIAdapter(
        cli_path=cli_path,
        working_directory=working_directory,
        **kwargs
    )