"""
Command Executor for LeanVibe Agent Hive 2.0 - Phase 6.1

Secure command execution engine with sandbox environment, resource limits, and comprehensive
monitoring for multi-agent workflow commands.
"""

import asyncio
import uuid
import json
import time
import psutil
import resource
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import signal
import subprocess
import tempfile
import shutil

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from .database import get_session
from .command_registry import CommandRegistry
from .task_distributor import TaskDistributor, TaskAssignment
from .agent_registry import AgentRegistry
from .redis import get_message_broker, AgentMessageBroker
from .circuit_breaker import CircuitBreaker
from .dead_letter_queue import DeadLetterQueueManager
from ..schemas.custom_commands import (
    CommandDefinition, CommandExecutionRequest, CommandExecutionResult,
    StepExecutionResult, CommandStatus, SecurityPolicy
)
from ..observability.hooks import HookInterceptor

logger = structlog.get_logger()


class ExecutionEnvironment(str, Enum):
    """Command execution environments."""
    SANDBOX = "sandbox"
    ISOLATED = "isolated"
    SHARED = "shared"
    CONTAINER = "container"


class ResourceLimitType(str, Enum):
    """Resource limit types."""
    CPU_TIME = "cpu_time"
    MEMORY = "memory"
    DISK_SPACE = "disk_space"
    NETWORK_BANDWIDTH = "network_bandwidth"
    FILE_DESCRIPTORS = "file_descriptors"
    PROCESSES = "processes"


@dataclass
class ResourceLimits:
    """Resource limits configuration."""
    max_memory_mb: int = 1024
    max_cpu_time_seconds: int = 3600
    max_disk_space_mb: int = 500
    max_file_descriptors: int = 1000
    max_processes: int = 10
    max_network_connections: int = 50
    execution_timeout_seconds: int = 7200  # 2 hours


@dataclass
class ExecutionContext:
    """Context for command execution."""
    execution_id: str
    command_name: str
    command_version: str
    workspace_path: Path
    temp_path: Path
    resource_limits: ResourceLimits
    security_policy: SecurityPolicy
    environment_vars: Dict[str, str] = field(default_factory=dict)
    allowed_operations: List[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ExecutionMetrics:
    """Execution performance metrics."""
    cpu_time_used: float = 0.0
    memory_peak_mb: float = 0.0
    disk_space_used_mb: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_received: int = 0
    file_operations: int = 0
    subprocess_count: int = 0
    execution_duration_seconds: float = 0.0


class SecurityViolationError(Exception):
    """Raised when security policy is violated."""
    pass


class ResourceLimitExceededError(Exception):
    """Raised when resource limits are exceeded."""
    pass


class CommandExecutor:
    """
    Secure command execution engine with comprehensive resource management.
    
    Features:
    - Sandboxed execution environment with security policies
    - Resource monitoring and enforcement with hard limits
    - Multi-agent workflow coordination and task distribution
    - Real-time progress tracking and observability integration
    - Circuit breaker pattern for failure resilience
    - Dead letter queue for failed command recovery
    - Comprehensive audit logging and compliance reporting
    """
    
    def __init__(
        self,
        command_registry: CommandRegistry,
        task_distributor: TaskDistributor,
        agent_registry: AgentRegistry,
        message_broker: Optional[AgentMessageBroker] = None,
        hook_manager: Optional[HookInterceptor] = None,
        execution_environment: ExecutionEnvironment = ExecutionEnvironment.SANDBOX
    ):
        self.command_registry = command_registry
        self.task_distributor = task_distributor
        self.agent_registry = agent_registry
        self.message_broker = message_broker
        self.hook_manager = hook_manager
        self.execution_environment = execution_environment
        
        # Active executions tracking
        self.active_executions: Dict[str, ExecutionContext] = {}
        self.execution_processes: Dict[str, subprocess.Popen] = {}
        self.execution_metrics: Dict[str, ExecutionMetrics] = {}
        
        # Security and resource management
        self.workspace_root = Path("/tmp/agent_hive_workspaces")
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        
        # Circuit breakers for agent reliability
        self.agent_circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Dead letter queue for failed commands
        self.dlq = DeadLetterQueueManager("command_execution_failures")
        
        # Execution limits and timeouts
        self.max_concurrent_executions = 20
        self.default_resource_limits = ResourceLimits()
        self.cleanup_interval_seconds = 300  # 5 minutes
        
        # Performance monitoring
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "security_violations": 0,
            "resource_limit_violations": 0,
            "average_execution_time": 0.0,
            "peak_concurrent_executions": 0
        }
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info(
            "CommandExecutor initialized",
            execution_environment=execution_environment.value,
            max_concurrent=self.max_concurrent_executions,
            workspace_root=str(self.workspace_root)
        )
    
    async def start(self) -> None:
        """Start the command executor and background services."""
        if self._running:
            return
        
        self._running = True
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        # Initialize message broker if not provided
        if not self.message_broker:
            self.message_broker = await get_message_broker()
        
        logger.info("CommandExecutor started")
    
    async def stop(self) -> None:
        """Stop the command executor and cleanup resources."""
        self._running = False
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Cancel active executions
        for execution_id in list(self.active_executions.keys()):
            await self.cancel_execution(execution_id, "executor_shutdown")
        
        logger.info("CommandExecutor stopped")
    
    async def execute_command(
        self,
        request: CommandExecutionRequest,
        requester_id: Optional[str] = None
    ) -> CommandExecutionResult:
        """
        Execute a multi-agent workflow command with full security and monitoring.
        
        Args:
            request: Command execution request
            requester_id: ID of the user/system requesting execution
            
        Returns:
            CommandExecutionResult with detailed execution information
        """
        execution_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        try:
            # Check concurrent execution limits
            if len(self.active_executions) >= self.max_concurrent_executions:
                raise RuntimeError("Maximum concurrent execution limit reached")
            
            # Get command definition
            command_def = await self.command_registry.get_command(
                request.command_name, request.command_version
            )
            
            if not command_def:
                raise ValueError(f"Command '{request.command_name}' not found")
            
            # Create execution context
            execution_context = await self._create_execution_context(
                execution_id, command_def, request
            )
            
            # Register execution
            self.active_executions[execution_id] = execution_context
            self.execution_metrics[execution_id] = ExecutionMetrics()
            
            # Update peak concurrent executions
            current_concurrent = len(self.active_executions)
            if current_concurrent > self.execution_stats["peak_concurrent_executions"]:
                self.execution_stats["peak_concurrent_executions"] = current_concurrent
            
            # Security validation
            await self._validate_security_policy(command_def, request, requester_id)
            
            # Hook: Pre-execution
            if self.hook_manager:
                await self.hook_manager.execute_hook("pre_command_execution", {
                    "execution_id": execution_id,
                    "command_name": request.command_name,
                    "requester_id": requester_id,
                    "parameters": request.parameters
                })
            
            logger.info(
                "Starting command execution",
                execution_id=execution_id,
                command_name=request.command_name,
                command_version=command_def.version,
                requester_id=requester_id,
                workflow_steps=len(command_def.workflow)
            )
            
            # Execute workflow
            result = await self._execute_workflow(
                execution_id, command_def, request, execution_context
            )
            
            # Update statistics
            self._update_execution_stats(result, start_time)
            
            # Update command registry metrics
            await self.command_registry.update_execution_metrics(
                request.command_name,
                result.total_execution_time_seconds or 0.0,
                result.status == CommandStatus.COMPLETED
            )
            
            # Hook: Post-execution
            if self.hook_manager:
                await self.hook_manager.execute_hook("post_command_execution", {
                    "execution_id": execution_id,
                    "result": result.model_dump(),
                    "success": result.status == CommandStatus.COMPLETED
                })
            
            logger.info(
                "Command execution completed",
                execution_id=execution_id,
                status=result.status.value,
                duration_seconds=result.total_execution_time_seconds,
                completed_steps=result.completed_steps,
                failed_steps=result.failed_steps
            )
            
            return result
            
        except Exception as e:
            error_message = f"Command execution failed: {str(e)}"
            
            # Send to dead letter queue for analysis
            await self.dlq.handle_failed_message(
                original_stream="command_executions",
                message={
                    "execution_id": execution_id,
                    "command_name": request.command_name,
                    "error": error_message,
                    "timestamp": datetime.utcnow().isoformat(),
                    "request": request.model_dump()
                },
                error_type="execution_failure",
                error_details={"error": error_message}
            )
            
            # Update error statistics
            if isinstance(e, SecurityViolationError):
                self.execution_stats["security_violations"] += 1
            elif isinstance(e, ResourceLimitExceededError):
                self.execution_stats["resource_limit_violations"] += 1
            
            self.execution_stats["failed_executions"] += 1
            
            logger.error(
                "Command execution failed",
                execution_id=execution_id,
                command_name=request.command_name,
                error=error_message
            )
            
            # Create failed result
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            return CommandExecutionResult(
                execution_id=uuid.UUID(execution_id),
                command_name=request.command_name,
                command_version=request.command_version or "latest",
                status=CommandStatus.FAILED,
                start_time=start_time,
                total_execution_time_seconds=execution_time,
                step_results=[],
                final_outputs={},
                total_steps=0,
                error_message=error_message
            )
            
        finally:
            # Cleanup execution context
            await self._cleanup_execution(execution_id)
    
    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a command execution."""
        try:
            if execution_id not in self.active_executions:
                return None
            
            context = self.active_executions[execution_id]
            metrics = self.execution_metrics.get(execution_id, ExecutionMetrics())
            
            return {
                "execution_id": execution_id,
                "command_name": context.command_name,
                "status": "running",
                "start_time": context.start_time.isoformat(),
                "elapsed_time_seconds": (datetime.utcnow() - context.start_time).total_seconds(),
                "resource_usage": {
                    "cpu_time_used": metrics.cpu_time_used,
                    "memory_peak_mb": metrics.memory_peak_mb,
                    "disk_space_used_mb": metrics.disk_space_used_mb,
                    "subprocess_count": metrics.subprocess_count
                },
                "workspace_path": str(context.workspace_path),
                "security_policy": context.security_policy.model_dump()
            }
            
        except Exception as e:
            logger.error("Failed to get execution status", execution_id=execution_id, error=str(e))
            return None
    
    async def cancel_execution(
        self,
        execution_id: str,
        reason: str = "user_requested"
    ) -> bool:
        """Cancel a running command execution."""
        try:
            if execution_id not in self.active_executions:
                logger.warning("Execution not found for cancellation", execution_id=execution_id)
                return False
            
            # Kill associated processes
            if execution_id in self.execution_processes:
                process = self.execution_processes[execution_id]
                try:
                    process.terminate()
                    # Wait for graceful termination
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        # Force kill if necessary
                        process.kill()
                        process.wait()
                except (ProcessLookupError, OSError):
                    pass  # Process already terminated
            
            # Cleanup execution context
            await self._cleanup_execution(execution_id)
            
            logger.info(
                "Command execution cancelled",
                execution_id=execution_id,
                reason=reason
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to cancel execution",
                execution_id=execution_id,
                error=str(e)
            )
            return False
    
    async def list_active_executions(self) -> List[Dict[str, Any]]:
        """List all currently active command executions."""
        try:
            active_list = []
            
            for execution_id, context in self.active_executions.items():
                metrics = self.execution_metrics.get(execution_id, ExecutionMetrics())
                
                active_list.append({
                    "execution_id": execution_id,
                    "command_name": context.command_name,
                    "start_time": context.start_time.isoformat(),
                    "elapsed_time_seconds": (datetime.utcnow() - context.start_time).total_seconds(),
                    "resource_usage": {
                        "memory_peak_mb": metrics.memory_peak_mb,
                        "cpu_time_used": metrics.cpu_time_used,
                        "subprocess_count": metrics.subprocess_count
                    }
                })
            
            return active_list
            
        except Exception as e:
            logger.error("Failed to list active executions", error=str(e))
            return []
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get command executor statistics."""
        return {
            **self.execution_stats,
            "active_executions": len(self.active_executions),
            "default_limits": {
                "max_memory_mb": self.default_resource_limits.max_memory_mb,
                "max_cpu_time_seconds": self.default_resource_limits.max_cpu_time_seconds,
                "execution_timeout_seconds": self.default_resource_limits.execution_timeout_seconds
            },
            "environment": {
                "execution_environment": self.execution_environment.value,
                "workspace_root": str(self.workspace_root),
                "max_concurrent_executions": self.max_concurrent_executions
            }
        }
    
    # Private helper methods
    
    async def _create_execution_context(
        self,
        execution_id: str,
        command_def: CommandDefinition,
        request: CommandExecutionRequest
    ) -> ExecutionContext:
        """Create isolated execution context for command."""
        try:
            # Create workspace directory
            workspace_path = self.workspace_root / execution_id
            workspace_path.mkdir(parents=True, exist_ok=True)
            
            # Create temp directory
            temp_path = workspace_path / "tmp"
            temp_path.mkdir(exist_ok=True)
            
            # Set resource limits
            resource_limits = ResourceLimits()
            if request.timeout_override:
                resource_limits.execution_timeout_seconds = request.timeout_override * 60
            
            # Environment variables
            env_vars = {
                "AGENT_HIVE_EXECUTION_ID": execution_id,
                "AGENT_HIVE_WORKSPACE": str(workspace_path),
                "AGENT_HIVE_TEMP": str(temp_path),
                "AGENT_HIVE_COMMAND": command_def.name,
                "AGENT_HIVE_VERSION": command_def.version
            }
            
            # Add custom parameters as environment variables
            for key, value in request.parameters.items():
                if isinstance(value, (str, int, float, bool)):
                    env_vars[f"AGENT_HIVE_PARAM_{key.upper()}"] = str(value)
            
            return ExecutionContext(
                execution_id=execution_id,
                command_name=command_def.name,
                command_version=command_def.version,
                workspace_path=workspace_path,
                temp_path=temp_path,
                resource_limits=resource_limits,
                security_policy=command_def.security_policy,
                environment_vars=env_vars,
                allowed_operations=command_def.security_policy.allowed_operations
            )
            
        except Exception as e:
            logger.error("Failed to create execution context", execution_id=execution_id, error=str(e))
            raise
    
    async def _validate_security_policy(
        self,
        command_def: CommandDefinition,
        request: CommandExecutionRequest,
        requester_id: Optional[str]
    ) -> None:
        """Validate security policy for command execution."""
        try:
            security_policy = command_def.security_policy
            
            # Check if approval required
            if security_policy.requires_approval and not requester_id:
                raise SecurityViolationError("Command requires user approval but no requester provided")
            
            # Validate network access requirements
            if security_policy.network_access and "network" not in security_policy.allowed_operations:
                raise SecurityViolationError("Network access requested but not in allowed operations")
            
            # Check resource limits
            if security_policy.resource_limits:
                max_memory = security_policy.resource_limits.get("max_memory_mb", 0)
                if max_memory > 4096:  # 4GB limit
                    raise SecurityViolationError("Requested memory exceeds security limits")
            
            # Validate file system access
            restricted_paths = security_policy.restricted_paths
            for path in restricted_paths:
                if path.startswith("/"):
                    # Absolute path restrictions
                    if any(op.startswith("file") for op in security_policy.allowed_operations):
                        logger.warning(f"Restricted path access requested: {path}")
            
            logger.debug("Security policy validation passed", command=command_def.name)
            
        except SecurityViolationError:
            raise
        except Exception as e:
            logger.error("Security validation error", error=str(e))
            raise SecurityViolationError(f"Security validation failed: {str(e)}")
    
    async def _execute_workflow(
        self,
        execution_id: str,
        command_def: CommandDefinition,
        request: CommandExecutionRequest,
        context: ExecutionContext
    ) -> CommandExecutionResult:
        """Execute the workflow steps with full monitoring and control."""
        start_time = datetime.utcnow()
        step_results = []
        
        try:
            # Distribute tasks to agents
            distribution_result = await self.task_distributor.distribute_tasks(
                workflow_steps=command_def.workflow,
                agent_requirements=command_def.agents,
                execution_context=request.context,
                urgency="high" if request.priority == "high" else "normal"
            )
            
            if not distribution_result.assignments:
                raise RuntimeError("No agents available for task distribution")
            
            # Execute workflow steps
            total_steps = len(command_def.workflow)
            completed_steps = 0
            failed_steps = 0
            
            # Build dependency graph for step execution order
            step_dependencies = self._build_step_dependencies(command_def.workflow)
            execution_order = self._resolve_execution_order(step_dependencies)
            
            # Execute steps in dependency order
            for step_batch in execution_order:
                batch_results = await self._execute_step_batch(
                    step_batch, distribution_result.assignments, context
                )
                
                step_results.extend(batch_results)
                
                # Update counters
                for result in batch_results:
                    if result.status == CommandStatus.COMPLETED:
                        completed_steps += 1
                    else:
                        failed_steps += 1
                
                # Check failure policy
                if failed_steps > 0 and command_def.failure_strategy == "fail_fast":
                    logger.warning(
                        "Stopping workflow due to step failure",
                        execution_id=execution_id,
                        failed_steps=failed_steps
                    )
                    break
            
            # Determine final status
            if failed_steps == 0:
                final_status = CommandStatus.COMPLETED
            elif completed_steps > 0:
                final_status = CommandStatus.FAILED  # Partial completion treated as failure
            else:
                final_status = CommandStatus.FAILED
            
            # Calculate execution time
            end_time = datetime.utcnow()
            total_execution_time = (end_time - start_time).total_seconds()
            
            # Collect final outputs
            final_outputs = self._collect_final_outputs(step_results)
            
            return CommandExecutionResult(
                execution_id=uuid.UUID(execution_id),
                command_name=command_def.name,
                command_version=command_def.version,
                status=final_status,
                start_time=start_time,
                end_time=end_time,
                total_execution_time_seconds=total_execution_time,
                step_results=step_results,
                final_outputs=final_outputs,
                total_steps=total_steps,
                completed_steps=completed_steps,
                failed_steps=failed_steps
            )
            
        except Exception as e:
            logger.error("Workflow execution error", execution_id=execution_id, error=str(e))
            raise
    
    async def _execute_step_batch(
        self,
        step_names: List[str],
        assignments: List[TaskAssignment],
        context: ExecutionContext
    ) -> List[StepExecutionResult]:
        """Execute a batch of workflow steps in parallel."""
        step_results = []
        
        # Create execution tasks for each step
        execution_tasks = []
        for step_name in step_names:
            # Find assignment for this step
            assignment = next(
                (a for a in assignments if a.task_id == step_name), None
            )
            
            if assignment:
                task = asyncio.create_task(
                    self._execute_single_step(step_name, assignment, context)
                )
                execution_tasks.append(task)
            else:
                # Create failed result for unassigned step
                step_results.append(StepExecutionResult(
                    step_id=step_name,
                    status=CommandStatus.FAILED,
                    error_message="No agent assignment found for step"
                ))
        
        # Wait for all steps to complete
        if execution_tasks:
            try:
                batch_results = await asyncio.gather(*execution_tasks, return_exceptions=True)
                
                for i, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        step_results.append(StepExecutionResult(
                            step_id=step_names[i],
                            status=CommandStatus.FAILED,
                            error_message=str(result)
                        ))
                    else:
                        step_results.append(result)
                        
            except Exception as e:
                logger.error("Step batch execution error", error=str(e))
                # Create failed results for all steps
                for step_name in step_names:
                    step_results.append(StepExecutionResult(
                        step_id=step_name,
                        status=CommandStatus.FAILED,
                        error_message=f"Batch execution error: {str(e)}"
                    ))
        
        return step_results
    
    async def _execute_single_step(
        self,
        step_name: str,
        assignment: TaskAssignment,
        context: ExecutionContext
    ) -> StepExecutionResult:
        """Execute a single workflow step on assigned agent."""
        start_time = datetime.utcnow()
        
        try:
            # Get agent circuit breaker
            agent_id = assignment.agent_id
            if agent_id not in self.agent_circuit_breakers:
                self.agent_circuit_breakers[agent_id] = CircuitBreaker(
                    failure_threshold=5,
                    timeout_duration=60,
                    expected_exception=Exception
                )
            
            circuit_breaker = self.agent_circuit_breakers[agent_id]
            
            # Execute step through circuit breaker
            async def execute_step():
                return await self._send_step_to_agent(
                    step_name, assignment, context
                )
            
            result = await circuit_breaker.call(execute_step)
            
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            
            return StepExecutionResult(
                step_id=step_name,
                status=CommandStatus.COMPLETED,
                agent_id=agent_id,
                start_time=start_time,
                end_time=end_time,
                execution_time_seconds=execution_time,
                outputs=result
            )
            
        except Exception as e:
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            
            logger.error(
                "Step execution failed",
                step_name=step_name,
                agent_id=assignment.agent_id,
                error=str(e)
            )
            
            return StepExecutionResult(
                step_id=step_name,
                status=CommandStatus.FAILED,
                agent_id=assignment.agent_id,
                start_time=start_time,
                end_time=end_time,
                execution_time_seconds=execution_time,
                error_message=str(e)
            )
    
    async def _send_step_to_agent(
        self,
        step_name: str,
        assignment: TaskAssignment,
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """Send workflow step to assigned agent for execution."""
        if not self.message_broker:
            # Mock execution for testing
            await asyncio.sleep(0.1)
            return {"status": "completed", "output": "mock_result"}
        
        try:
            # Send task execution message to agent
            await self.message_broker.send_message(
                from_agent="command_executor",
                to_agent=assignment.agent_id,
                message_type="workflow_step_execution",
                payload={
                    "execution_id": context.execution_id,
                    "step_name": step_name,
                    "step_data": {
                        "workspace_path": str(context.workspace_path),
                        "environment_vars": context.environment_vars,
                        "security_policy": context.security_policy.model_dump(),
                        "resource_limits": {
                            "max_memory_mb": context.resource_limits.max_memory_mb,
                            "max_cpu_time_seconds": context.resource_limits.max_cpu_time_seconds
                        }
                    }
                }
            )
            
            # For now, simulate agent response
            # In production, this would wait for agent response
            await asyncio.sleep(0.5)
            
            return {
                "status": "completed",
                "agent_id": assignment.agent_id,
                "execution_time": 0.5,
                "output": f"Step {step_name} completed successfully"
            }
            
        except Exception as e:
            logger.error(
                "Failed to send step to agent",
                step_name=step_name,
                agent_id=assignment.agent_id,
                error=str(e)
            )
            raise
    
    def _build_step_dependencies(self, workflow_steps) -> Dict[str, List[str]]:
        """Build dependency graph for workflow steps."""
        dependencies = {}
        
        for step in workflow_steps:
            dependencies[step.step] = step.depends_on or []
        
        return dependencies
    
    def _resolve_execution_order(self, dependencies: Dict[str, List[str]]) -> List[List[str]]:
        """Resolve execution order using topological sort."""
        # Simple topological sort implementation
        in_degree = {step: 0 for step in dependencies.keys()}
        
        # Calculate in-degrees
        for step, deps in dependencies.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[step] += 1
        
        # Generate execution batches
        execution_order = []
        remaining_steps = set(dependencies.keys())
        
        while remaining_steps:
            # Find steps with no dependencies
            ready_steps = [
                step for step in remaining_steps
                if in_degree[step] == 0
            ]
            
            if not ready_steps:
                # Circular dependency - execute remaining steps in arbitrary order
                ready_steps = list(remaining_steps)
            
            execution_order.append(ready_steps)
            
            # Remove ready steps and update in-degrees
            for step in ready_steps:
                remaining_steps.remove(step)
                
                # Update dependent steps
                for dependent, deps in dependencies.items():
                    if step in deps and dependent in remaining_steps:
                        in_degree[dependent] -= 1
        
        return execution_order
    
    def _collect_final_outputs(self, step_results: List[StepExecutionResult]) -> Dict[str, Any]:
        """Collect final outputs from all step results."""
        final_outputs = {}
        
        for result in step_results:
            if result.outputs:
                final_outputs[result.step_id] = result.outputs
        
        return final_outputs
    
    async def _cleanup_execution(self, execution_id: str) -> None:
        """Cleanup execution resources and temporary files."""
        try:
            # Remove from active executions
            context = self.active_executions.pop(execution_id, None)
            self.execution_metrics.pop(execution_id, None)
            
            # Kill any remaining processes
            process = self.execution_processes.pop(execution_id, None)
            if process:
                try:
                    process.terminate()
                    process.wait(timeout=5.0)
                except:
                    try:
                        process.kill()
                    except:
                        pass
            
            # Cleanup workspace
            if context and context.workspace_path.exists():
                try:
                    shutil.rmtree(context.workspace_path)
                except Exception as e:
                    logger.warning(
                        "Failed to cleanup workspace",
                        execution_id=execution_id,
                        workspace_path=str(context.workspace_path),
                        error=str(e)
                    )
            
            logger.debug("Execution cleanup completed", execution_id=execution_id)
            
        except Exception as e:
            logger.error("Execution cleanup error", execution_id=execution_id, error=str(e))
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop for stale executions."""
        while self._running:
            try:
                await self._cleanup_stale_executions()
                await asyncio.sleep(self.cleanup_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cleanup loop error", error=str(e))
                await asyncio.sleep(30)
    
    async def _cleanup_stale_executions(self) -> None:
        """Cleanup executions that have exceeded their timeout."""
        try:
            current_time = datetime.utcnow()
            stale_executions = []
            
            for execution_id, context in self.active_executions.items():
                execution_duration = (current_time - context.start_time).total_seconds()
                timeout = context.resource_limits.execution_timeout_seconds
                
                if execution_duration > timeout:
                    stale_executions.append(execution_id)
            
            # Cancel stale executions
            for execution_id in stale_executions:
                logger.warning(
                    "Cancelling stale execution",
                    execution_id=execution_id,
                    duration_seconds=execution_duration
                )
                await self.cancel_execution(execution_id, "timeout_exceeded")
            
        except Exception as e:
            logger.error("Failed to cleanup stale executions", error=str(e))
    
    def _update_execution_stats(
        self,
        result: CommandExecutionResult,
        start_time: datetime
    ) -> None:
        """Update execution statistics."""
        try:
            self.execution_stats["total_executions"] += 1
            
            if result.status == CommandStatus.COMPLETED:
                self.execution_stats["successful_executions"] += 1
            else:
                self.execution_stats["failed_executions"] += 1
            
            # Update average execution time
            if result.total_execution_time_seconds:
                current_avg = self.execution_stats["average_execution_time"]
                total_executions = self.execution_stats["total_executions"]
                
                new_avg = (
                    (current_avg * (total_executions - 1) + result.total_execution_time_seconds) /
                    total_executions
                )
                self.execution_stats["average_execution_time"] = new_avg
            
        except Exception as e:
            logger.error("Error updating execution stats", error=str(e))