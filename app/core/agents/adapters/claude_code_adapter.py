"""
Claude Code Adapter for Multi-CLI Coordination

This adapter enables Claude Code CLI to participate in coordinated multi-agent
workflows. It implements the UniversalAgentInterface to provide standardized
communication, task execution, and capability reporting.

Key Features:
- Subprocess-based Claude Code CLI integration
- Secure worktree isolation and path validation
- Message format translation (universal ↔ Claude Code)
- Performance monitoring and error handling
- Capability-based task routing support

Implementation Status: TEMPLATE - Requires completion
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
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
# Claude Code Specific Models
# ================================================================================

@dataclass
class ClaudeCodeCommand:
    """Claude Code CLI command specification"""
    command: str
    args: List[str] = field(default_factory=list)
    options: Dict[str, str] = field(default_factory=dict)
    input_files: List[str] = field(default_factory=list)
    output_format: str = "json"
    timeout_seconds: int = 300
    
    def to_cli_args(self) -> List[str]:
        """Convert to CLI argument list"""
        args = [self.command]
        
        # Add input files
        args.extend(self.input_files)
        
        # Add options
        for key, value in self.options.items():
            if key.startswith('--'):
                args.append(key)
                if value:
                    args.append(value)
            else:
                args.append(f"--{key}")
                if value:
                    args.append(value)
        
        # Add positional args
        args.extend(self.args)
        
        return args

@dataclass
class ClaudeCodeResponse:
    """Claude Code CLI response structure"""
    success: bool
    output: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time: float = 0.0
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""

# ================================================================================
# Claude Code Adapter Implementation
# ================================================================================

class ClaudeCodeAdapter(UniversalAgentInterface):
    """
    Adapter for Claude Code CLI integration in multi-agent workflows.
    
    This adapter translates universal task formats to Claude Code CLI commands,
    executes them with proper isolation, and translates responses back to
    universal formats for seamless coordination.
    
    Capabilities:
    - Code analysis and review
    - Documentation generation
    - Refactoring and optimization
    - Test generation
    - Architecture design
    - Debugging assistance
    
    Security Features:
    - Worktree-based isolation
    - Path traversal prevention
    - Resource usage monitoring
    - Command injection protection
    
    Performance Features:
    - Concurrent task execution
    - Resource usage tracking
    - Performance metrics collection
    - Intelligent caching
    """
    
    def __init__(self, agent_id: str, agent_type: AgentType = AgentType.CLAUDE_CODE):
        """
        Initialize Claude Code adapter.
        
        Args:
            agent_id: Unique identifier for this agent instance
            agent_type: Agent type (should be CLAUDE_CODE)
        """
        super().__init__(agent_id, agent_type)
        
        # Configuration
        self._cli_path: str = "claude"  # Default CLI command
        self._working_directory: str = ""
        self._max_concurrent_tasks: int = 3
        self._default_timeout: int = 300
        self._resource_limits: Dict[str, float] = {
            "max_cpu_percent": 80.0,
            "max_memory_mb": 1024.0,
            "max_execution_time": 3600
        }
        
        # State tracking
        self._is_initialized: bool = False
        self._capabilities_cache: Optional[List[AgentCapability]] = None
        self._last_health_check: Optional[datetime] = None
        self._performance_metrics: Dict[str, float] = {}
        self._active_processes: Dict[str, subprocess.Popen] = {}
        
        # Security settings
        self._allowed_commands: Set[str] = {
            "analyze", "review", "implement", "refactor", "test", 
            "document", "debug", "optimize", "format"
        }
        self._restricted_paths: List[str] = [
            "/etc", "/usr", "/bin", "/sbin", "/root", "/sys", "/proc"
        ]
    
    # ================================================================================
    # Core Task Execution - REQUIRES IMPLEMENTATION
    # ================================================================================
    
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """
        Execute a task using Claude Code CLI.
        
        IMPLEMENTATION REQUIRED:
        This method needs to be completed to:
        1. Validate task and security constraints
        2. Translate universal task to Claude Code format
        3. Execute CLI command with proper isolation
        4. Monitor execution and resource usage
        5. Translate response back to universal format
        
        Args:
            task: Universal task to execute
            
        Returns:
            AgentResult: Standardized execution result
            
        TODO: Complete implementation following this structure:
        1. Validate task against capabilities and security
        2. Create Claude Code command from task
        3. Execute with subprocess in isolated environment
        4. Monitor performance and resource usage
        5. Parse results and create AgentResult
        """
        # TODO: Implement task execution
        # This is the critical method that needs to be implemented
        
        # Start timing
        start_time = time.time()
        
        # Create result object
        result = AgentResult(
            task_id=task.id,
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            status=TaskStatus.PENDING
        )
        result.mark_started()
        
        try:
            # 1. Validate task
            self._validate_task(task)
            
            # 2. Translate task to Claude Code command
            command = self._translate_task_to_command(task)
            
            # 3. Execute command with isolation
            response = await self._execute_claude_command(command, task.context)
            
            # 4. Process results
            if response.success:
                result.output_data = response.output
                result.files_created = response.files_created
                result.files_modified = response.files_modified
                result.mark_completed(success=True)
            else:
                result.error_message = response.error_message
                result.mark_completed(success=False)
            
            # 5. Add performance metrics
            result.execution_time_seconds = time.time() - start_time
            result.resource_usage = await self._get_resource_usage()
            
            return result
            
        except Exception as e:
            logger.error(f"Task execution failed for {task.id}: {e}")
            result.error_message = str(e)
            result.mark_completed(success=False)
            return result
    
    def _translate_task_to_command(self, task: AgentTask) -> ClaudeCodeCommand:
        """
        Translate universal task to Claude Code CLI command.
        
        IMPLEMENTATION REQUIRED:
        This method needs to map task types to appropriate Claude Code commands
        and format the arguments correctly.
        
        Args:
            task: Universal task to translate
            
        Returns:
            ClaudeCodeCommand: Claude Code specific command
            
        TODO: Implement translation logic for each capability type
        """
        # TODO: Implement task-to-command translation
        # Map CapabilityType to Claude Code commands
        
        command_map = {
            CapabilityType.CODE_ANALYSIS: "analyze",
            CapabilityType.CODE_REVIEW: "review", 
            CapabilityType.CODE_IMPLEMENTATION: "implement",
            CapabilityType.REFACTORING: "refactor",
            CapabilityType.DOCUMENTATION: "document",
            CapabilityType.TESTING: "test",
            CapabilityType.DEBUGGING: "debug"
        }
        
        base_command = command_map.get(task.type, "analyze")
        
        # Build command based on task requirements
        command = ClaudeCodeCommand(
            command=base_command,
            timeout_seconds=task.timeout_seconds
        )
        
        # Add context-specific arguments
        if task.context and task.context.file_scope:
            command.input_files = task.context.file_scope
        
        # Add task-specific options
        if task.description:
            command.options["--description"] = task.description
        
        if task.requirements:
            command.options["--requirements"] = ",".join(task.requirements)
        
        return command
    
    async def _execute_claude_command(
        self,
        command: ClaudeCodeCommand,
        context: Optional[ExecutionContext]
    ) -> ClaudeCodeResponse:
        """
        Execute Claude Code CLI command with proper isolation.
        
        IMPLEMENTATION REQUIRED:
        This method needs to:
        1. Set up isolated execution environment
        2. Execute subprocess with security constraints
        3. Monitor resource usage
        4. Parse and validate output
        
        Args:
            command: Claude Code command to execute
            context: Execution context with isolation settings
            
        Returns:
            ClaudeCodeResponse: Command execution result
            
        TODO: Implement secure subprocess execution
        """
        # TODO: Implement secure command execution
        
        start_time = time.time()
        
        try:
            # 1. Prepare execution environment
            work_dir = context.worktree_path if context else self._working_directory
            env = self._prepare_environment(context)
            
            # 2. Build command arguments
            cli_args = [self._cli_path] + command.to_cli_args()
            
            # 3. Execute subprocess
            process = await asyncio.create_subprocess_exec(
                *cli_args,
                cwd=work_dir,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=1024*1024  # 1MB output limit
            )
            
            # 4. Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=command.timeout_seconds
                )
            except asyncio.TimeoutError:
                process.kill()
                raise TaskExecutionError("Command execution timed out")
            
            # 5. Parse output
            stdout_str = stdout.decode('utf-8') if stdout else ""
            stderr_str = stderr.decode('utf-8') if stderr else ""
            
            success = process.returncode == 0
            output = {}
            
            if success and stdout_str:
                try:
                    output = json.loads(stdout_str)
                except json.JSONDecodeError:
                    output = {"raw_output": stdout_str}
            
            return ClaudeCodeResponse(
                success=success,
                output=output,
                error_message=stderr_str if not success else None,
                execution_time=time.time() - start_time,
                stdout=stdout_str,
                stderr=stderr_str
            )
            
        except Exception as e:
            return ClaudeCodeResponse(
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    def _prepare_environment(self, context: Optional[ExecutionContext]) -> Dict[str, str]:
        """
        Prepare execution environment with security constraints.
        
        IMPLEMENTATION REQUIRED:
        Set up environment variables while maintaining security.
        
        Args:
            context: Execution context
            
        Returns:
            Dict[str, str]: Environment variables
        """
        # TODO: Implement environment preparation
        env = os.environ.copy()
        
        # Add context-specific environment variables
        if context and context.environment_variables:
            for key, value in context.environment_variables.items():
                # Validate environment variables for security
                if self._is_safe_env_var(key, value):
                    env[key] = value
        
        # Set working directory
        if context and context.worktree_path:
            env['CLAUDE_WORK_DIR'] = context.worktree_path
        
        return env
    
    def _is_safe_env_var(self, key: str, value: str) -> bool:
        """Validate environment variable for security."""
        # TODO: Implement security validation
        unsafe_keys = ['PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH']
        return key not in unsafe_keys and not key.startswith('SUDO_')
    
    # ================================================================================
    # Capabilities Reporting - REQUIRES IMPLEMENTATION
    # ================================================================================
    
    async def get_capabilities(self) -> List[AgentCapability]:
        """
        Report Claude Code capabilities.
        
        IMPLEMENTATION REQUIRED:
        This method should return accurate capability assessments based on
        Claude Code's actual abilities and current system state.
        
        Returns:
            List[AgentCapability]: Agent capabilities with confidence scores
            
        TODO: Implement dynamic capability assessment
        """
        # TODO: Implement capability reporting
        # This should reflect Claude Code's actual capabilities
        
        if self._capabilities_cache:
            return self._capabilities_cache
        
        # Define Claude Code's core capabilities
        capabilities = [
            AgentCapability(
                type=CapabilityType.CODE_ANALYSIS,
                confidence=0.95,
                performance_score=0.90,
                estimated_time_seconds=60
            ),
            AgentCapability(
                type=CapabilityType.CODE_REVIEW,
                confidence=0.92,
                performance_score=0.88,
                estimated_time_seconds=120
            ),
            AgentCapability(
                type=CapabilityType.DOCUMENTATION,
                confidence=0.90,
                performance_score=0.85,
                estimated_time_seconds=180
            ),
            AgentCapability(
                type=CapabilityType.REFACTORING,
                confidence=0.85,
                performance_score=0.80,
                estimated_time_seconds=300
            ),
            AgentCapability(
                type=CapabilityType.DEBUGGING,
                confidence=0.88,
                performance_score=0.83,
                estimated_time_seconds=240
            ),
            AgentCapability(
                type=CapabilityType.TESTING,
                confidence=0.80,
                performance_score=0.75,
                estimated_time_seconds=200
            ),
            AgentCapability(
                type=CapabilityType.ARCHITECTURE_DESIGN,
                confidence=0.85,
                performance_score=0.80,
                estimated_time_seconds=600
            )
        ]
        
        # TODO: Adjust capabilities based on current system state
        # - Check CLI tool availability
        # - Assess current load
        # - Update confidence based on recent performance
        
        self._capabilities_cache = capabilities
        return capabilities
    
    # ================================================================================
    # Health Monitoring - REQUIRES IMPLEMENTATION
    # ================================================================================
    
    async def health_check(self) -> HealthStatus:
        """
        Perform comprehensive health check.
        
        IMPLEMENTATION REQUIRED:
        This method should:
        1. Check Claude Code CLI availability
        2. Measure response times
        3. Assess resource usage
        4. Evaluate current capacity
        
        Returns:
            HealthStatus: Comprehensive health information
            
        TODO: Implement health monitoring
        """
        # TODO: Implement health checking
        
        start_time = time.time()
        
        try:
            # 1. Check CLI availability
            cli_available = await self._check_cli_availability()
            
            # 2. Measure response time
            response_time = (time.time() - start_time) * 1000  # milliseconds
            
            # 3. Get system metrics
            cpu_usage = await self._get_cpu_usage()
            memory_usage = await self._get_memory_usage()
            
            # 4. Calculate health state
            health_state = self._determine_health_state(
                cli_available, response_time, cpu_usage, memory_usage
            )
            
            # 5. Get performance metrics
            metrics = await self.get_performance_metrics()
            
            return HealthStatus(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                state=health_state,
                response_time_ms=response_time,
                cpu_usage_percent=cpu_usage,
                memory_usage_mb=memory_usage,
                active_tasks=len(self._current_tasks),
                completed_tasks=len(self._task_history),
                failed_tasks=sum(1 for r in self._task_history if r.status == TaskStatus.FAILED),
                last_activity=datetime.utcnow(),
                error_rate=metrics.get("error_rate", 0.0),
                throughput_tasks_per_minute=metrics.get("throughput", 0.0),
                uptime_seconds=metrics.get("uptime_seconds", 0.0)
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthStatus(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                state=HealthState.UNHEALTHY,
                response_time_ms=0.0,
                cpu_usage_percent=0.0,
                memory_usage_mb=0.0,
                active_tasks=0,
                completed_tasks=0,
                failed_tasks=0,
                last_activity=datetime.utcnow(),
                error_rate=1.0,
                throughput_tasks_per_minute=0.0,
                uptime_seconds=0.0
            )
    
    async def _check_cli_availability(self) -> bool:
        """Check if Claude Code CLI is available and responsive."""
        # TODO: Implement CLI availability check
        try:
            process = await asyncio.create_subprocess_exec(
                self._cli_path, "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=10
            )
            return process.returncode == 0
        except Exception:
            return False
    
    async def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        # TODO: Implement CPU monitoring
        return 0.0
    
    async def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        # TODO: Implement memory monitoring
        return 0.0
    
    async def _get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage metrics."""
        # TODO: Implement resource monitoring
        return {
            "cpu_percent": await self._get_cpu_usage(),
            "memory_mb": await self._get_memory_usage(),
            "active_processes": len(self._active_processes)
        }
    
    def _determine_health_state(
        self,
        cli_available: bool,
        response_time: float,
        cpu_usage: float,
        memory_usage: float
    ) -> HealthState:
        """Determine overall health state based on metrics."""
        if not cli_available:
            return HealthState.OFFLINE
        
        if response_time > 5000 or cpu_usage > 90 or memory_usage > 2048:
            return HealthState.UNHEALTHY
        
        if response_time > 2000 or cpu_usage > 70 or memory_usage > 1024:
            return HealthState.DEGRADED
        
        return HealthState.HEALTHY
    
    # ================================================================================
    # Lifecycle Management - REQUIRES IMPLEMENTATION
    # ================================================================================
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize Claude Code adapter.
        
        IMPLEMENTATION REQUIRED:
        This method should:
        1. Validate configuration
        2. Check CLI tool availability
        3. Set up working directories
        4. Initialize monitoring
        
        Args:
            config: Configuration parameters
            
        Returns:
            bool: True if initialization successful
            
        TODO: Implement initialization logic
        """
        # TODO: Implement initialization
        
        try:
            # 1. Load configuration
            self._cli_path = config.get("cli_path", "claude")
            self._working_directory = config.get("working_directory", "/tmp")
            self._max_concurrent_tasks = config.get("max_concurrent_tasks", 3)
            self._default_timeout = config.get("default_timeout", 300)
            
            # 2. Validate CLI availability
            if not await self._check_cli_availability():
                logger.error("Claude Code CLI not available")
                return False
            
            # 3. Create working directory if needed
            os.makedirs(self._working_directory, exist_ok=True)
            
            # 4. Initialize monitoring
            self._start_time = datetime.utcnow()
            
            self._is_initialized = True
            logger.info(f"Claude Code adapter {self.agent_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Claude Code adapter initialization failed: {e}")
            return False
    
    async def shutdown(self) -> None:
        """
        Gracefully shutdown Claude Code adapter.
        
        IMPLEMENTATION REQUIRED:
        This method should:
        1. Cancel active tasks
        2. Clean up resources
        3. Save state if needed
        
        TODO: Implement shutdown logic
        """
        # TODO: Implement shutdown
        
        try:
            # 1. Cancel active tasks
            for task_id in list(self._current_tasks.keys()):
                await self.cancel_task(task_id)
            
            # 2. Terminate active processes
            for process in self._active_processes.values():
                try:
                    process.terminate()
                    await asyncio.sleep(1)
                    if process.poll() is None:
                        process.kill()
                except Exception as e:
                    logger.error(f"Error terminating process: {e}")
            
            # 3. Clear state
            self._active_processes.clear()
            self._current_tasks.clear()
            self._is_initialized = False
            
            logger.info(f"Claude Code adapter {self.agent_id} shutdown completed")
            
        except Exception as e:
            logger.error(f"Claude Code adapter shutdown error: {e}")

# ================================================================================
# Factory Functions and Utilities
# ================================================================================

def create_claude_code_adapter(
    agent_id: str,
    cli_path: str = "claude",
    working_directory: str = "/tmp"
) -> ClaudeCodeAdapter:
    """
    Factory function to create Claude Code adapter with configuration.
    
    Args:
        agent_id: Unique identifier for the adapter
        cli_path: Path to Claude Code CLI executable
        working_directory: Working directory for task execution
        
    Returns:
        ClaudeCodeAdapter: Configured adapter instance
    """
    adapter = ClaudeCodeAdapter(agent_id)
    return adapter

def validate_claude_code_environment() -> Dict[str, bool]:
    """
    Validate Claude Code environment and dependencies.
    
    Returns:
        Dict[str, bool]: Validation results
    """
    results = {
        "cli_available": False,
        "python_available": False,
        "git_available": False,
        "working_directory_writable": False
    }
    
    # TODO: Implement environment validation
    # Check CLI availability, dependencies, permissions, etc.
    
    return results

# ================================================================================
# Constants and Configuration
# ================================================================================

# Default Claude Code commands
CLAUDE_CODE_COMMANDS = {
    CapabilityType.CODE_ANALYSIS: "analyze",
    CapabilityType.CODE_REVIEW: "review",
    CapabilityType.CODE_IMPLEMENTATION: "implement",
    CapabilityType.REFACTORING: "refactor",
    CapabilityType.DOCUMENTATION: "document",
    CapabilityType.TESTING: "test",
    CapabilityType.DEBUGGING: "debug",
    CapabilityType.ARCHITECTURE_DESIGN: "design"
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "response_time_warning": 2000,  # ms
    "response_time_critical": 5000,  # ms
    "cpu_usage_warning": 70.0,  # %
    "cpu_usage_critical": 90.0,  # %
    "memory_usage_warning": 1024.0,  # MB
    "memory_usage_critical": 2048.0  # MB
}

# Security settings
SECURITY_SETTINGS = {
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "allowed_file_extensions": [".py", ".js", ".ts", ".md", ".txt", ".json", ".yaml"],
    "blocked_commands": ["rm", "del", "format", "sudo", "su"],
    "max_execution_time": 3600  # 1 hour
}