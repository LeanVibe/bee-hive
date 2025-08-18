"""
Cursor CLI Adapter for Multi-CLI Coordination

This adapter enables Cursor CLI to participate in coordinated multi-agent
workflows. It implements the UniversalAgentInterface to provide standardized
communication, task execution, and capability reporting.

Key Features:
- Subprocess-based Cursor CLI integration
- Secure worktree isolation and path validation
- Message format translation (universal ↔ Cursor)
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
# CURSOR CLI COMMAND MODEL
# ================================================================================

@dataclass
class CursorCommand:
    """Represents a Cursor CLI command with all parameters."""
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
class CursorResponse:
    """Represents the response from Cursor CLI execution."""
    success: bool
    output: str
    error_output: str
    return_code: int
    execution_time: float
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    memory_usage_mb: Optional[float] = None


# ================================================================================
# CURSOR CLI ADAPTER IMPLEMENTATION
# ================================================================================

class CursorAdapter(UniversalAgentInterface):
    """
    Production Cursor CLI Adapter.
    
    Provides seamless integration with Cursor CLI through the universal
    agent interface, enabling Cursor to participate in multi-agent workflows.
    """
    
    def __init__(
        self,
        cli_path: str = "cursor",
        working_directory: Optional[str] = None,
        max_concurrent_tasks: int = 3,
        default_timeout: float = 300.0
    ):
        """
        Initialize Cursor adapter.
        
        Args:
            cli_path: Path to Cursor CLI executable
            working_directory: Default working directory for operations
            max_concurrent_tasks: Maximum concurrent task limit
            default_timeout: Default timeout for operations
        """
        agent_id = f"cursor_agent_{str(uuid.uuid4())[:8]}"
        super().__init__(agent_id, AgentType.CURSOR)
        
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
        
        logger.info(f"Cursor adapter initialized: {agent_id}")

    # ========================================================================
    # UNIVERSAL AGENT INTERFACE IMPLEMENTATION  
    # ========================================================================

    async def execute_task(self, task: AgentTask) -> AgentResult:
        """
        Execute a task using Cursor CLI.
        
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
                    error_message=f"Task type {task.type} not supported by Cursor"
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
            
            # 3. Translate universal task to Cursor command
            cursor_command = self._translate_task_to_command(task)
            
            # 4. Execute Cursor command
            response = await self._execute_cursor_command(cursor_command, task.context)
            
            # 5. Translate response to universal format
            result = self._translate_response_to_result(task.id, response)
            
            # 6. Update performance metrics
            execution_time = time.time() - start_time
            self._total_execution_time += execution_time
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
        Get Cursor agent capabilities.
        
        Returns:
            List of agent capabilities with confidence scores
        """
        return [
            AgentCapability(
                type=CapabilityType.CODE_IMPLEMENTATION,
                confidence=0.95,
                performance_score=0.95,
                description="Code generation and implementation with AI assistance"
            ),
            AgentCapability(
                type=CapabilityType.REFACTORING,
                confidence=0.90,
                performance_score=0.90,
                description="Code refactoring and optimization"
            ),
            AgentCapability(
                type=CapabilityType.CODE_REVIEW,
                confidence=0.85,
                performance_score=0.85,
                description="Code review and quality analysis"
            ),
            AgentCapability(
                type=CapabilityType.TESTING,
                confidence=0.80,
                performance_score=0.80,
                description="Test generation and testing workflows"
            ),
            AgentCapability(
                type=CapabilityType.DEBUGGING,
                confidence=0.85,
                performance_score=0.85,
                description="Debugging and error resolution"
            ),
            AgentCapability(
                type=CapabilityType.DOCUMENTATION,
                confidence=0.75,
                performance_score=0.75,
                description="Code documentation generation"
            ),
            AgentCapability(
                type=CapabilityType.CODE_ANALYSIS,
                confidence=0.80,
                performance_score=0.80,
                description="Static code analysis and metrics"
            )
        ]

    async def health_check(self) -> HealthStatus:
        """
        Perform health check on Cursor adapter.
        
        Returns:
            HealthStatus: Current health status
        """
        start_time = time.time()
        
        try:
            # Test Cursor CLI availability
            test_process = await asyncio.create_subprocess_exec(
                self._cli_path, "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(
                test_process.communicate(),
                timeout=10.0
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
                    cpu_usage_percent=15.0,  # Estimated based on Cursor resource usage
                    memory_usage_mb=256.0,   # Estimated based on Cursor resource usage
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
                    error_message="Cursor CLI not responding properly",
                    last_activity=self._last_activity
                )
                
        except asyncio.TimeoutError:
            return HealthStatus(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                state=HealthState.UNHEALTHY,
                response_time_ms=10000.0,
                error_message="Cursor CLI health check timeout",
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
    # CURSOR-SPECIFIC IMPLEMENTATION
    # ========================================================================

    async def _is_task_compatible(self, task: AgentTask) -> bool:
        """Check if task is compatible with Cursor capabilities."""
        compatible_types = {
            CapabilityType.CODE_IMPLEMENTATION,
            CapabilityType.REFACTORING,
            CapabilityType.CODE_REVIEW,
            CapabilityType.TESTING,
            CapabilityType.DEBUGGING,
            CapabilityType.DOCUMENTATION,
            CapabilityType.CODE_ANALYSIS
        }
        return task.type in compatible_types

    def _translate_task_to_command(self, task: AgentTask) -> CursorCommand:
        """
        Translate universal task to Cursor CLI command.
        
        Args:
            task: Universal agent task
            
        Returns:
            CursorCommand: Cursor-specific command
        """
        # Map capability types to Cursor CLI commands
        command_map = {
            CapabilityType.CODE_IMPLEMENTATION: "generate",
            CapabilityType.REFACTORING: "refactor", 
            CapabilityType.CODE_REVIEW: "review",
            CapabilityType.TESTING: "test",
            CapabilityType.DEBUGGING: "debug",
            CapabilityType.DOCUMENTATION: "document",
            CapabilityType.CODE_ANALYSIS: "analyze"
        }
        
        base_command = command_map.get(task.type, "generate")
        
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
        
        # Get input files
        input_files = []
        if task.context and task.context.file_scope:
            input_files = task.context.file_scope
        elif task.input_data and "files" in task.input_data:
            input_files = task.input_data["files"]
        
        command = CursorCommand(
            command=base_command,
            options=options,
            input_files=input_files,
            timeout_seconds=task.timeout_seconds or self._default_timeout
        )
        
        return command

    async def _execute_cursor_command(
        self,
        command: CursorCommand,
        context: Optional[ExecutionContext]
    ) -> CursorResponse:
        """
        Execute Cursor CLI command with proper isolation.
        
        Args:
            command: Cursor command to execute
            context: Execution context with isolation settings
            
        Returns:
            CursorResponse: Command execution result
        """
        start_time = time.time()
        
        try:
            # 1. Prepare execution environment
            work_dir = context.worktree_path if context else self._working_directory
            if not work_dir or not os.path.exists(work_dir):
                work_dir = tempfile.mkdtemp(prefix="cursor_work_")
                logger.info(f"Created temporary work directory: {work_dir}")
            
            env = self._prepare_environment(context)
            
            # 2. Validate command safety
            self._validate_command_safety(command)
            
            # 3. Build command arguments
            cli_args = [self._cli_path] + command.to_cli_args()
            logger.debug(f"Executing Cursor command: {' '.join(cli_args)}")
            
            # 4. Track files before execution
            existing_files = set()
            if os.path.exists(work_dir):
                for root, dirs, files in os.walk(work_dir):
                    for file in files:
                        existing_files.add(os.path.join(root, file))
            
            # 5. Execute subprocess with timeout
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
                    f"Cursor command timed out after {command.timeout_seconds}s"
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
            
            return CursorResponse(
                success=process.returncode == 0,
                output=stdout.decode('utf-8') if stdout else "",
                error_output=stderr.decode('utf-8') if stderr else "",
                return_code=process.returncode,
                execution_time=execution_time,
                files_created=files_created,
                files_modified=files_modified
            )
            
        except Exception as e:
            logger.error(f"Cursor command execution failed: {e}")
            execution_time = time.time() - start_time
            
            return CursorResponse(
                success=False,
                output="",
                error_output=str(e),
                return_code=-1,
                execution_time=execution_time
            )

    def _translate_response_to_result(
        self,
        task_id: str,
        response: CursorResponse
    ) -> AgentResult:
        """
        Translate Cursor response to universal agent result.
        
        Args:
            task_id: Original task ID
            response: Cursor command response
            
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
            
            # Add file change information
            output_data.update({
                "files_created": response.files_created,
                "files_modified": response.files_modified,
                "execution_time": response.execution_time
            })
            
            return AgentResult(
                task_id=task_id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                status=TaskStatus.COMPLETED,
                output_data=output_data,
                execution_time=response.execution_time,
                metadata={
                    "cursor_return_code": response.return_code,
                    "memory_usage_mb": response.memory_usage_mb
                }
            )
        else:
            return AgentResult(
                task_id=task_id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                status=TaskStatus.FAILED,
                error_message=response.error_output or "Cursor command failed",
                execution_time=response.execution_time,
                metadata={
                    "cursor_return_code": response.return_code,
                    "raw_output": response.output
                }
            )

    def _prepare_environment(self, context: Optional[ExecutionContext]) -> Dict[str, str]:
        """Prepare environment variables for Cursor execution."""
        env = os.environ.copy()
        
        if context and context.environment_vars:
            env.update(context.environment_vars)
        
        # Add Cursor-specific environment variables
        env["CURSOR_OUTPUT_FORMAT"] = "json"
        env["CURSOR_DISABLE_TELEMETRY"] = "true"
        
        return env

    def _validate_command_safety(self, command: CursorCommand):
        """Validate command for security and safety."""
        # Check for dangerous commands
        dangerous_patterns = [
            "rm -rf", "sudo", "chmod 777", "curl", "wget",
            "eval", "exec", "system", "__import__"
        ]
        
        command_str = " ".join(command.to_cli_args()).lower()
        
        for pattern in dangerous_patterns:
            if pattern in command_str:
                raise SecurityError(f"Potentially dangerous command pattern detected: {pattern}")
        
        # Validate file paths
        for file_path in command.input_files:
            if not self._is_safe_path(file_path):
                raise SecurityError(f"Unsafe file path: {file_path}")

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
        Initialize the Cursor adapter with configuration.
        
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
            
            # Verify Cursor CLI is available
            health = await self.health_check()
            
            if health.state == HealthState.HEALTHY:
                logger.info(f"Cursor adapter {self.agent_id} initialized successfully")
                return True
            else:
                logger.error(f"Cursor adapter initialization failed: {health.error_message}")
                return False
                
        except Exception as e:
            logger.error(f"Cursor adapter initialization error: {e}")
            return False

    async def shutdown(self) -> None:
        """
        Gracefully shutdown the Cursor adapter.
        """
        logger.info(f"Shutting down Cursor adapter {self.agent_id}")
        
        # Wait for active tasks to complete (with timeout)
        shutdown_timeout = 30.0
        start_time = time.time()
        
        while self._active_tasks and (time.time() - start_time) < shutdown_timeout:
            logger.info(f"Waiting for {len(self._active_tasks)} active tasks to complete...")
            await asyncio.sleep(1.0)
        
        if self._active_tasks:
            logger.warning(f"Shutdown timeout reached, {len(self._active_tasks)} tasks may be incomplete")
        
        logger.info(f"Cursor adapter {self.agent_id} shutdown completed")


# ================================================================================
# FACTORY FUNCTION
# ================================================================================

def create_cursor_adapter(
    cli_path: str = "cursor",
    working_directory: Optional[str] = None,
    **kwargs
) -> CursorAdapter:
    """
    Factory function to create a Cursor adapter instance.
    
    Args:
        cli_path: Path to Cursor CLI executable
        working_directory: Working directory for operations
        **kwargs: Additional configuration options
        
    Returns:
        CursorAdapter: Configured Cursor adapter instance
    """
    return CursorAdapter(
        cli_path=cli_path,
        working_directory=working_directory,
        **kwargs
    )