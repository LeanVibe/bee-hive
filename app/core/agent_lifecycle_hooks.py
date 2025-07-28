"""
Agent Lifecycle Hooks for LeanVibe Agent Hive 2.0

Python-based hook system specifically designed for agent lifecycle events.
Provides PreToolUse/PostToolUse hooks with security validation, performance
monitoring, and real-time event streaming.
"""

import asyncio
import uuid
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import json

import structlog
from sqlalchemy import select, insert
from sqlalchemy.ext.asyncio import AsyncSession

from .database import get_async_session
from .redis import get_redis
from .hook_lifecycle_system import HookLifecycleSystem, HookEvent, HookType, SecurityRisk, DangerousCommand
from .agent_messaging_service import AgentMessagingService, MessageType, MessagePriority
from ..models.observability import AgentEvent, EventType

logger = structlog.get_logger()


class ToolExecutionPhase(str, Enum):
    """Phases of tool execution."""
    PRE_VALIDATION = "pre_validation"
    PRE_EXECUTION = "pre_execution"
    EXECUTION = "execution"
    POST_EXECUTION = "post_execution"
    POST_VALIDATION = "post_validation"


class SecurityAction(str, Enum):
    """Security actions for dangerous commands."""
    ALLOW = "allow"
    BLOCK = "block"
    REQUIRE_APPROVAL = "require_approval"
    LOG_AND_CONTINUE = "log_and_continue"


@dataclass
class ToolExecutionContext:
    """Context for tool execution hooks."""
    agent_id: uuid.UUID
    session_id: Optional[uuid.UUID]
    tool_name: str
    parameters: Dict[str, Any]
    phase: ToolExecutionPhase
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    security_context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": str(self.agent_id),
            "session_id": str(self.session_id) if self.session_id else None,
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "phase": self.phase.value,
            "execution_id": self.execution_id,
            "started_at": self.started_at.isoformat(),
            "metadata": self.metadata,
            "security_context": self.security_context
        }


@dataclass
class HookExecutionResult:
    """Result of hook execution."""
    success: bool
    execution_time_ms: float
    security_action: SecurityAction = SecurityAction.ALLOW
    blocked_reason: Optional[str] = None
    modifications: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "execution_time_ms": self.execution_time_ms,
            "security_action": self.security_action.value,
            "blocked_reason": self.blocked_reason,
            "modifications": self.modifications,
            "metadata": self.metadata
        }


class AgentLifecycleHooks:
    """
    Python-based hook system for agent lifecycle events.
    
    Provides comprehensive PreToolUse/PostToolUse hooks with security
    validation, performance monitoring, and integration with the
    agent messaging system.
    """
    
    def __init__(
        self,
        redis_client=None,
        messaging_service: Optional[AgentMessagingService] = None,
        hook_system: Optional[HookLifecycleSystem] = None
    ):
        self.redis = redis_client or get_redis()
        self.messaging_service = messaging_service
        self.hook_system = hook_system
        
        # Hook registry
        self.pre_tool_hooks: List[Callable] = []
        self.post_tool_hooks: List[Callable] = []
        self.error_hooks: List[Callable] = []
        
        # Security configuration
        self.dangerous_commands = self._initialize_dangerous_commands()
        self.blocked_commands: Set[str] = set()
        self.security_enabled = True
        
        # Performance tracking
        self.hook_execution_times: Dict[str, List[float]] = {}
        self.security_violations: List[Dict[str, Any]] = []
        self.hook_call_counts: Dict[str, int] = {}
        
        # Active executions
        self.active_executions: Dict[str, ToolExecutionContext] = {}
        
        logger.info("🪝 Agent Lifecycle Hooks initialized")
    
    def register_pre_tool_hook(self, hook_func: Callable) -> None:
        """Register a PreToolUse hook function."""
        self.pre_tool_hooks.append(hook_func)
        logger.info(f"📝 Registered PreToolUse hook: {hook_func.__name__}")
    
    def register_post_tool_hook(self, hook_func: Callable) -> None:
        """Register a PostToolUse hook function."""
        self.post_tool_hooks.append(hook_func)
        logger.info(f"📝 Registered PostToolUse hook: {hook_func.__name__}")
    
    def register_error_hook(self, hook_func: Callable) -> None:
        """Register an error handling hook function."""
        self.error_hooks.append(hook_func)
        logger.info(f"📝 Registered error hook: {hook_func.__name__}")
    
    async def execute_pre_tool_hooks(
        self,
        agent_id: uuid.UUID,
        session_id: Optional[uuid.UUID],
        tool_name: str,
        parameters: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> HookExecutionResult:
        """
        Execute PreToolUse hooks with security validation.
        
        Args:
            agent_id: ID of the executing agent
            session_id: Session ID if applicable
            tool_name: Name of the tool being executed
            parameters: Tool parameters
            metadata: Additional metadata
        
        Returns:
            HookExecutionResult with execution details
        """
        start_time = time.time()
        execution_id = str(uuid.uuid4())
        
        context = ToolExecutionContext(
            agent_id=agent_id,
            session_id=session_id,
            tool_name=tool_name,
            parameters=parameters,
            phase=ToolExecutionPhase.PRE_VALIDATION,
            execution_id=execution_id,
            metadata=metadata or {}
        )
        
        self.active_executions[execution_id] = context
        
        try:
            # Security validation first
            security_result = await self._validate_security(context)
            if security_result.security_action == SecurityAction.BLOCK:
                await self._log_security_violation(context, security_result.blocked_reason)
                return security_result
            
            context.phase = ToolExecutionPhase.PRE_EXECUTION
            
            # Execute registered PreToolUse hooks
            modifications = {}
            for hook_func in self.pre_tool_hooks:
                try:
                    hook_start = time.time()
                    
                    if asyncio.iscoroutinefunction(hook_func):
                        result = await hook_func(context)
                    else:
                        result = hook_func(context)
                    
                    hook_time = (time.time() - hook_start) * 1000
                    self._record_hook_performance(hook_func.__name__, hook_time)
                    
                    # Merge any modifications from hooks
                    if isinstance(result, dict):
                        modifications.update(result)
                        
                except Exception as e:
                    logger.error(f"PreToolUse hook failed: {hook_func.__name__}", error=str(e))
                    await self._execute_error_hooks(context, e)
            
            # Create hook event for the hook system
            if self.hook_system:
                hook_event = HookEvent(
                    hook_type=HookType.PRE_TOOL_USE,
                    agent_id=agent_id,
                    session_id=session_id,
                    timestamp=datetime.utcnow(),
                    payload={
                        "tool_name": tool_name,
                        "parameters": parameters,
                        "execution_id": execution_id,
                        "security_action": security_result.security_action.value,
                        "modifications": modifications
                    }
                )
                await self.hook_system.process_hook_event(hook_event)
            
            # Send hook message via messaging service
            if self.messaging_service:
                await self.messaging_service.send_lifecycle_message(
                    message_type=MessageType.HOOK_PRE_TOOL_USE,
                    from_agent="hook_system",
                    to_agent=str(agent_id),
                    payload={
                        "tool_name": tool_name,
                        "execution_id": execution_id,
                        "security_action": security_result.security_action.value,
                        "modifications": modifications,
                        "context": context.to_dict()
                    },
                    priority=MessagePriority.HIGH
                )
            
            # Store in database
            await self._store_hook_event(
                agent_id=agent_id,
                event_type=EventType.PRE_TOOL_USE,
                tool_name=tool_name,
                execution_id=execution_id,
                payload={
                    "parameters": parameters,
                    "security_action": security_result.security_action.value,
                    "modifications": modifications
                }
            )
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            result = HookExecutionResult(
                success=True,
                execution_time_ms=execution_time_ms,
                security_action=security_result.security_action,
                blocked_reason=security_result.blocked_reason,
                modifications=modifications
            )
            
            logger.info(
                "✅ PreToolUse hooks executed",
                agent_id=str(agent_id),
                tool_name=tool_name,
                execution_id=execution_id,
                execution_time_ms=execution_time_ms,
                security_action=security_result.security_action.value
            )
            
            return result
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(
                "❌ PreToolUse hooks failed",
                agent_id=str(agent_id),
                tool_name=tool_name,
                execution_id=execution_id,
                error=str(e)
            )
            
            await self._execute_error_hooks(context, e)
            
            return HookExecutionResult(
                success=False,
                execution_time_ms=execution_time_ms,
                security_action=SecurityAction.BLOCK,
                blocked_reason=f"Hook execution failed: {str(e)}"
            )
        
        finally:
            # Clean up active execution
            self.active_executions.pop(execution_id, None)
    
    async def execute_post_tool_hooks(
        self,
        agent_id: uuid.UUID,
        session_id: Optional[uuid.UUID],
        tool_name: str,
        parameters: Dict[str, Any],
        result: Dict[str, Any],
        success: bool,
        execution_time_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> HookExecutionResult:
        """
        Execute PostToolUse hooks with result processing.
        
        Args:
            agent_id: ID of the executing agent
            session_id: Session ID if applicable
            tool_name: Name of the executed tool
            parameters: Tool parameters used
            result: Tool execution result
            success: Whether tool execution was successful
            execution_time_ms: Tool execution time
            metadata: Additional metadata
        
        Returns:
            HookExecutionResult with execution details
        """
        start_time = time.time()
        execution_id = str(uuid.uuid4())
        
        context = ToolExecutionContext(
            agent_id=agent_id,
            session_id=session_id,
            tool_name=tool_name,
            parameters=parameters,
            phase=ToolExecutionPhase.POST_EXECUTION,
            execution_id=execution_id,
            metadata={
                **(metadata or {}),
                "tool_result": result,
                "tool_success": success,
                "tool_execution_time_ms": execution_time_ms
            }
        )
        
        try:
            context.phase = ToolExecutionPhase.POST_VALIDATION
            
            # Execute registered PostToolUse hooks
            modifications = {}
            for hook_func in self.post_tool_hooks:
                try:
                    hook_start = time.time()
                    
                    if asyncio.iscoroutinefunction(hook_func):
                        hook_result = await hook_func(context)
                    else:
                        hook_result = hook_func(context)
                    
                    hook_time = (time.time() - hook_start) * 1000
                    self._record_hook_performance(hook_func.__name__, hook_time)
                    
                    # Merge any modifications from hooks
                    if isinstance(hook_result, dict):
                        modifications.update(hook_result)
                        
                except Exception as e:
                    logger.error(f"PostToolUse hook failed: {hook_func.__name__}", error=str(e))
                    await self._execute_error_hooks(context, e)
            
            # Create hook event for the hook system
            if self.hook_system:
                hook_event = HookEvent(
                    hook_type=HookType.POST_TOOL_USE,
                    agent_id=agent_id,
                    session_id=session_id,
                    timestamp=datetime.utcnow(),
                    payload={
                        "tool_name": tool_name,
                        "parameters": parameters,
                        "result": result,
                        "success": success,
                        "execution_time_ms": execution_time_ms,
                        "execution_id": execution_id,
                        "modifications": modifications
                    }
                )
                await self.hook_system.process_hook_event(hook_event)
            
            # Send hook message via messaging service
            if self.messaging_service:
                await self.messaging_service.send_lifecycle_message(
                    message_type=MessageType.HOOK_POST_TOOL_USE,
                    from_agent="hook_system",
                    to_agent=str(agent_id),
                    payload={
                        "tool_name": tool_name,
                        "execution_id": execution_id,
                        "success": success,
                        "execution_time_ms": execution_time_ms,
                        "result": result,
                        "modifications": modifications,
                        "context": context.to_dict()
                    },
                    priority=MessagePriority.NORMAL
                )
            
            # Store in database
            await self._store_hook_event(
                agent_id=agent_id,
                event_type=EventType.POST_TOOL_USE,
                tool_name=tool_name,
                execution_id=execution_id,
                payload={
                    "parameters": parameters,
                    "result": result,
                    "success": success,
                    "tool_execution_time_ms": execution_time_ms,
                    "modifications": modifications
                }
            )
            
            hook_execution_time_ms = (time.time() - start_time) * 1000
            
            result_obj = HookExecutionResult(
                success=True,
                execution_time_ms=hook_execution_time_ms,
                modifications=modifications,
                metadata={
                    "tool_success": success,
                    "tool_execution_time_ms": execution_time_ms
                }
            )
            
            logger.info(
                "✅ PostToolUse hooks executed",
                agent_id=str(agent_id),
                tool_name=tool_name,
                execution_id=execution_id,
                hook_execution_time_ms=hook_execution_time_ms,
                tool_success=success
            )
            
            return result_obj
            
        except Exception as e:
            hook_execution_time_ms = (time.time() - start_time) * 1000
            logger.error(
                "❌ PostToolUse hooks failed",
                agent_id=str(agent_id),
                tool_name=tool_name,
                execution_id=execution_id,
                error=str(e)
            )
            
            await self._execute_error_hooks(context, e)
            
            return HookExecutionResult(
                success=False,
                execution_time_ms=hook_execution_time_ms,
                blocked_reason=f"Hook execution failed: {str(e)}"
            )
    
    async def get_hook_metrics(self) -> Dict[str, Any]:
        """Get hook system performance metrics."""
        # Calculate average execution times
        avg_execution_times = {}
        for hook_name, times in self.hook_execution_times.items():
            if times:
                avg_execution_times[hook_name] = {
                    "count": len(times),
                    "average_ms": sum(times) / len(times),
                    "min_ms": min(times),
                    "max_ms": max(times)
                }
        
        return {
            "registered_hooks": {
                "pre_tool_hooks": len(self.pre_tool_hooks),
                "post_tool_hooks": len(self.post_tool_hooks),
                "error_hooks": len(self.error_hooks)
            },
            "security": {
                "security_enabled": self.security_enabled,
                "dangerous_commands_count": len(self.dangerous_commands),
                "blocked_commands_count": len(self.blocked_commands),
                "security_violations_count": len(self.security_violations)
            },
            "performance": {
                "hook_call_counts": dict(self.hook_call_counts),
                "average_execution_times": avg_execution_times,
                "active_executions": len(self.active_executions)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _validate_security(self, context: ToolExecutionContext) -> HookExecutionResult:
        """Validate security for tool execution."""
        if not self.security_enabled:
            return HookExecutionResult(
                success=True,
                execution_time_ms=0,
                security_action=SecurityAction.ALLOW
            )
        
        # Check for dangerous commands
        tool_command = f"{context.tool_name} {json.dumps(context.parameters)}"
        
        for dangerous_cmd in self.dangerous_commands:
            if dangerous_cmd.matches(tool_command):
                if dangerous_cmd.block_execution:
                    return HookExecutionResult(
                        success=False,
                        execution_time_ms=0,
                        security_action=SecurityAction.BLOCK,
                        blocked_reason=f"Dangerous command detected: {dangerous_cmd.description}"
                    )
                elif dangerous_cmd.require_approval:
                    return HookExecutionResult(
                        success=True,
                        execution_time_ms=0,
                        security_action=SecurityAction.REQUIRE_APPROVAL,
                        blocked_reason=f"Command requires approval: {dangerous_cmd.description}"
                    )
        
        # Check explicitly blocked commands
        if context.tool_name in self.blocked_commands:
            return HookExecutionResult(
                success=False,
                execution_time_ms=0,
                security_action=SecurityAction.BLOCK,
                blocked_reason=f"Tool '{context.tool_name}' is explicitly blocked"
            )
        
        return HookExecutionResult(
            success=True,
            execution_time_ms=0,
            security_action=SecurityAction.ALLOW
        )
    
    async def _execute_error_hooks(self, context: ToolExecutionContext, error: Exception) -> None:
        """Execute error handling hooks."""
        for hook_func in self.error_hooks:
            try:
                if asyncio.iscoroutinefunction(hook_func):
                    await hook_func(context, error)
                else:
                    hook_func(context, error)
            except Exception as e:
                logger.error(f"Error hook failed: {hook_func.__name__}", error=str(e))
    
    async def _log_security_violation(self, context: ToolExecutionContext, reason: str) -> None:
        """Log security violation."""
        violation = {
            "agent_id": str(context.agent_id),
            "tool_name": context.tool_name,
            "parameters": context.parameters,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
            "execution_id": context.execution_id
        }
        
        self.security_violations.append(violation)
        
        # Send security alert via messaging
        if self.messaging_service:
            await self.messaging_service.send_lifecycle_message(
                message_type=MessageType.HOOK_ERROR,
                from_agent="security_system",
                to_agent="orchestrator",
                payload={
                    "violation_type": "dangerous_command",
                    "agent_id": str(context.agent_id),
                    "tool_name": context.tool_name,
                    "reason": reason,
                    "severity": "high"
                },
                priority=MessagePriority.CRITICAL
            )
        
        logger.warning(
            "🚨 Security violation detected",
            agent_id=str(context.agent_id),
            tool_name=context.tool_name,
            reason=reason,
            execution_id=context.execution_id
        )
    
    async def _store_hook_event(
        self,
        agent_id: uuid.UUID,
        event_type: EventType,
        tool_name: str,
        execution_id: str,
        payload: Dict[str, Any]
    ) -> None:
        """Store hook event in database."""
        try:
            async with get_async_session() as db:
                event = AgentEvent(
                    agent_id=agent_id,
                    event_type=event_type,
                    payload={
                        **payload,
                        "tool_name": tool_name,
                        "execution_id": execution_id
                    }
                )
                db.add(event)
                await db.commit()
        except Exception as e:
            logger.error("Failed to store hook event", error=str(e))
    
    def _record_hook_performance(self, hook_name: str, execution_time_ms: float) -> None:
        """Record hook performance metrics."""
        if hook_name not in self.hook_execution_times:
            self.hook_execution_times[hook_name] = []
        
        self.hook_execution_times[hook_name].append(execution_time_ms)
        self.hook_call_counts[hook_name] = self.hook_call_counts.get(hook_name, 0) + 1
        
        # Keep only last 1000 measurements per hook
        if len(self.hook_execution_times[hook_name]) > 1000:
            self.hook_execution_times[hook_name] = self.hook_execution_times[hook_name][-1000:]
    
    def _initialize_dangerous_commands(self) -> List[DangerousCommand]:
        """Initialize list of dangerous commands to monitor."""
        return [
            DangerousCommand(
                pattern=r"rm\s+-rf\s+/",
                risk_level=SecurityRisk.CRITICAL,
                description="Recursive deletion of root directory",
                block_execution=True
            ),
            DangerousCommand(
                pattern=r"sudo\s+(rm|rmdir|del)",
                risk_level=SecurityRisk.HIGH,
                description="Privileged file deletion",
                block_execution=True
            ),
            DangerousCommand(
                pattern=r"(curl|wget).*\|(sh|bash|python)",
                risk_level=SecurityRisk.HIGH,
                description="Download and execute remote script",
                block_execution=True
            ),
            DangerousCommand(
                pattern=r"chmod\s+777",
                risk_level=SecurityRisk.MEDIUM,
                description="Setting overly permissive file permissions",
                require_approval=True
            ),
            DangerousCommand(
                pattern=r"docker\s+run.*--privileged",
                risk_level=SecurityRisk.HIGH,
                description="Running privileged Docker container",
                require_approval=True
            ),
            DangerousCommand(
                pattern=r"git\s+push.*--force",
                risk_level=SecurityRisk.MEDIUM,
                description="Force pushing to Git repository",
                require_approval=True
            )
        ]