"""
Orchestrator Hook Integration for LeanVibe Agent Hive 2.0.

This module provides seamless integration between the HookLifecycleSystem
and the AgentOrchestrator, enabling comprehensive lifecycle event tracking
throughout the orchestration process.

Features:
- Automatic hook event generation for orchestrator operations
- Agent lifecycle tracking with hook integration
- Performance monitoring and security validation
- Real-time dashboard updates via WebSocket streaming
- Redis integration for event persistence and analysis
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from contextlib import asynccontextmanager
import time

import structlog

from .hook_lifecycle_system import (
    HookLifecycleSystem,
    HookType,
    HookProcessingResult,
    get_hook_lifecycle_system,
    process_pre_tool_use_hook,
    process_post_tool_use_hook,
    process_stop_hook,
    process_notification_hook
)
from .orchestrator import AgentOrchestrator, AgentInstance, AgentRole, AgentStatus
from ..models.agent import Agent, AgentType

logger = structlog.get_logger()


class OrchestratorHookIntegration:
    """
    Integration layer between AgentOrchestrator and HookLifecycleSystem.
    
    Provides automatic hook event generation for orchestrator operations
    with comprehensive lifecycle tracking and security validation.
    """
    
    def __init__(self, orchestrator: AgentOrchestrator):
        self.orchestrator = orchestrator
        self.hook_system: Optional[HookLifecycleSystem] = None
        
        # Integration state
        self.integration_enabled = True
        self.hook_processors: Dict[str, List[Callable]] = {}
        
        # Performance tracking
        self.metrics = {
            "hooks_generated": 0,
            "hooks_processed": 0,
            "hooks_failed": 0,
            "agent_lifecycle_events": 0,
            "task_execution_events": 0,
            "avg_hook_processing_time_ms": 0.0
        }
        
        # Configuration
        self.config = {
            "track_agent_lifecycle": True,
            "track_task_execution": True,
            "track_inter_agent_communication": True,
            "track_performance_metrics": True,
            "enable_security_validation": True,
            "generate_stop_events": True
        }
    
    async def initialize(self) -> None:
        """Initialize the orchestrator hook integration."""
        try:
            # Get the hook lifecycle system
            self.hook_system = await get_hook_lifecycle_system()
            
            # Register default hook processors
            await self._register_default_hook_processors()
            
            # Patch orchestrator methods for hook integration
            await self._patch_orchestrator_methods()
            
            logger.info("🔗 Orchestrator Hook Integration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Orchestrator Hook Integration: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the orchestrator hook integration."""
        try:
            # Restore original orchestrator methods
            await self._restore_orchestrator_methods()
            
            logger.info("🔗 Orchestrator Hook Integration shutdown completed")
            
        except Exception as e:
            logger.error(f"Orchestrator Hook Integration shutdown error: {e}")
    
    async def _register_default_hook_processors(self) -> None:
        """Register default hook processors for orchestrator events."""
        
        # Agent lifecycle processor
        async def process_agent_lifecycle_event(event_data: Dict[str, Any]) -> None:
            """Process agent lifecycle events."""
            try:
                agent_id = event_data.get("agent_id")
                event_type = event_data.get("event_type")
                
                logger.info(
                    f"Processing agent lifecycle event",
                    agent_id=agent_id,
                    event_type=event_type
                )
                
                self.metrics["agent_lifecycle_events"] += 1
                
            except Exception as e:
                logger.error(f"Agent lifecycle event processing error: {e}")
        
        # Task execution processor
        async def process_task_execution_event(event_data: Dict[str, Any]) -> None:
            """Process task execution events."""
            try:
                task_id = event_data.get("task_id")
                agent_id = event_data.get("agent_id")
                status = event_data.get("status")
                
                logger.info(
                    f"Processing task execution event",
                    task_id=task_id,
                    agent_id=agent_id,
                    status=status
                )
                
                self.metrics["task_execution_events"] += 1
                
            except Exception as e:
                logger.error(f"Task execution event processing error: {e}")
        
        # Register processors
        self.hook_processors["agent_lifecycle"] = [process_agent_lifecycle_event]
        self.hook_processors["task_execution"] = [process_task_execution_event]
    
    async def _patch_orchestrator_methods(self) -> None:
        """Patch orchestrator methods to integrate hook generation."""
        
        # Store original methods
        self._original_spawn_agent = self.orchestrator.spawn_agent
        self._original_shutdown_agent = self.orchestrator.shutdown_agent
        self._original_execute_task = getattr(self.orchestrator, 'execute_task', None)
        
        # Patch spawn_agent method
        async def hooked_spawn_agent(
            role: AgentRole,
            agent_id: Optional[str] = None,
            capabilities: Optional[List] = None,
            **kwargs
        ) -> str:
            """Hooked spawn_agent method with lifecycle event generation."""
            start_time = time.time()
            
            try:
                # Generate PreToolUse hook
                if self.config["track_agent_lifecycle"] and self.hook_system:
                    await self.hook_system.process_hook(
                        hook_type=HookType.PRE_TOOL_USE,
                        agent_id=uuid.uuid4(),  # Temporary ID for orchestrator
                        session_id=None,
                        payload={
                            "tool_name": "spawn_agent",
                            "parameters": {
                                "role": role.value,
                                "agent_id": agent_id,
                                "capabilities": [str(cap) for cap in (capabilities or [])]
                            },
                            "operation_type": "agent_lifecycle"
                        },
                        priority=2  # High priority for lifecycle events
                    )
                
                # Execute original method
                result_agent_id = await self._original_spawn_agent(role, agent_id, capabilities, **kwargs)
                
                # Generate PostToolUse hook
                execution_time = (time.time() - start_time) * 1000
                if self.config["track_agent_lifecycle"] and self.hook_system:
                    await self.hook_system.process_hook(
                        hook_type=HookType.POST_TOOL_USE,
                        agent_id=uuid.UUID(result_agent_id),
                        session_id=None,
                        payload={
                            "tool_name": "spawn_agent",
                            "success": True,
                            "result": {"agent_id": result_agent_id, "role": role.value},
                            "execution_time_ms": execution_time,
                            "operation_type": "agent_lifecycle"
                        },
                        priority=2
                    )
                
                # Generate agent start notification
                if self.config["track_agent_lifecycle"] and self.hook_system:
                    await self.hook_system.process_notification(
                        agent_id=uuid.UUID(result_agent_id),
                        session_id=None,
                        level="info",
                        message=f"Agent {result_agent_id} started with role {role.value}",
                        details={
                            "agent_id": result_agent_id,
                            "role": role.value,
                            "capabilities": [str(cap) for cap in (capabilities or [])],
                            "spawn_time_ms": execution_time
                        }
                    )
                
                self.metrics["hooks_generated"] += 3  # PreToolUse, PostToolUse, Notification
                return result_agent_id
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                
                # Generate error PostToolUse hook
                if self.config["track_agent_lifecycle"] and self.hook_system:
                    await self.hook_system.process_hook(
                        hook_type=HookType.POST_TOOL_USE,
                        agent_id=uuid.uuid4(),  # Temporary ID for orchestrator
                        session_id=None,
                        payload={
                            "tool_name": "spawn_agent",
                            "success": False,
                            "error": str(e),
                            "execution_time_ms": execution_time,
                            "operation_type": "agent_lifecycle"
                        },
                        priority=1  # Very high priority for errors
                    )
                
                self.metrics["hooks_failed"] += 1
                raise
        
        # Patch shutdown_agent method
        async def hooked_shutdown_agent(
            agent_id: str,
            graceful: bool = True,
            **kwargs
        ) -> None:
            """Hooked shutdown_agent method with lifecycle event generation."""
            start_time = time.time()
            
            try:
                # Generate PreToolUse hook
                if self.config["track_agent_lifecycle"] and self.hook_system:
                    await self.hook_system.process_hook(
                        hook_type=HookType.PRE_TOOL_USE,
                        agent_id=uuid.UUID(agent_id),
                        session_id=None,
                        payload={
                            "tool_name": "shutdown_agent",
                            "parameters": {
                                "agent_id": agent_id,
                                "graceful": graceful
                            },
                            "operation_type": "agent_lifecycle"
                        },
                        priority=2
                    )
                
                # Execute original method
                await self._original_shutdown_agent(agent_id, graceful, **kwargs)
                
                # Generate PostToolUse hook
                execution_time = (time.time() - start_time) * 1000
                if self.config["track_agent_lifecycle"] and self.hook_system:
                    await self.hook_system.process_hook(
                        hook_type=HookType.POST_TOOL_USE,
                        agent_id=uuid.UUID(agent_id),
                        session_id=None,
                        payload={
                            "tool_name": "shutdown_agent",
                            "success": True,
                            "result": {"agent_id": agent_id, "graceful": graceful},
                            "execution_time_ms": execution_time,
                            "operation_type": "agent_lifecycle"
                        },
                        priority=2
                    )
                
                # Generate stop event
                if self.config["generate_stop_events"] and self.hook_system:
                    await self.hook_system.process_stop(
                        agent_id=uuid.UUID(agent_id),
                        session_id=None,
                        reason="Agent shutdown requested",
                        details={
                            "graceful": graceful,
                            "shutdown_time_ms": execution_time
                        }
                    )
                
                self.metrics["hooks_generated"] += 3  # PreToolUse, PostToolUse, Stop
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                
                # Generate error PostToolUse hook
                if self.config["track_agent_lifecycle"] and self.hook_system:
                    await self.hook_system.process_hook(
                        hook_type=HookType.POST_TOOL_USE,
                        agent_id=uuid.UUID(agent_id),
                        session_id=None,
                        payload={
                            "tool_name": "shutdown_agent",
                            "success": False,
                            "error": str(e),
                            "execution_time_ms": execution_time,
                            "operation_type": "agent_lifecycle"
                        },
                        priority=1
                    )
                
                self.metrics["hooks_failed"] += 1
                raise
        
        # Apply patches
        self.orchestrator.spawn_agent = hooked_spawn_agent
        self.orchestrator.shutdown_agent = hooked_shutdown_agent
        
        # Patch execute_task if it exists
        if self._original_execute_task:
            async def hooked_execute_task(task_data: Dict[str, Any], **kwargs) -> Any:
                """Hooked execute_task method with task execution tracking."""
                start_time = time.time()
                task_id = task_data.get("task_id", str(uuid.uuid4()))
                agent_id = task_data.get("agent_id")
                
                try:
                    # Generate PreToolUse hook
                    if self.config["track_task_execution"] and self.hook_system:
                        await self.hook_system.process_hook(
                            hook_type=HookType.PRE_TOOL_USE,
                            agent_id=uuid.UUID(agent_id) if agent_id else uuid.uuid4(),
                            session_id=task_data.get("session_id"),
                            payload={
                                "tool_name": "execute_task",
                                "parameters": {
                                    "task_id": task_id,
                                    "task_type": task_data.get("task_type"),
                                    "priority": task_data.get("priority", 5)
                                },
                                "operation_type": "task_execution"
                            },
                            correlation_id=task_id,
                            priority=3
                        )
                    
                    # Execute original method
                    result = await self._original_execute_task(task_data, **kwargs)
                    
                    # Generate PostToolUse hook
                    execution_time = (time.time() - start_time) * 1000
                    if self.config["track_task_execution"] and self.hook_system:
                        await self.hook_system.process_hook(
                            hook_type=HookType.POST_TOOL_USE,
                            agent_id=uuid.UUID(agent_id) if agent_id else uuid.uuid4(),
                            session_id=task_data.get("session_id"),
                            payload={
                                "tool_name": "execute_task",
                                "success": True,
                                "result": {"task_id": task_id, "result_summary": str(result)[:500]},
                                "execution_time_ms": execution_time,
                                "operation_type": "task_execution"
                            },
                            correlation_id=task_id,
                            priority=3
                        )
                    
                    self.metrics["hooks_generated"] += 2
                    return result
                    
                except Exception as e:
                    execution_time = (time.time() - start_time) * 1000
                    
                    # Generate error PostToolUse hook
                    if self.config["track_task_execution"] and self.hook_system:
                        await self.hook_system.process_hook(
                            hook_type=HookType.POST_TOOL_USE,
                            agent_id=uuid.UUID(agent_id) if agent_id else uuid.uuid4(),
                            session_id=task_data.get("session_id"),
                            payload={
                                "tool_name": "execute_task",
                                "success": False,
                                "error": str(e),
                                "execution_time_ms": execution_time,
                                "operation_type": "task_execution"
                            },
                            correlation_id=task_id,
                            priority=1
                        )
                    
                    self.metrics["hooks_failed"] += 1
                    raise
            
            self.orchestrator.execute_task = hooked_execute_task
    
    async def _restore_orchestrator_methods(self) -> None:
        """Restore original orchestrator methods."""
        if hasattr(self, '_original_spawn_agent'):
            self.orchestrator.spawn_agent = self._original_spawn_agent
        
        if hasattr(self, '_original_shutdown_agent'):
            self.orchestrator.shutdown_agent = self._original_shutdown_agent
        
        if hasattr(self, '_original_execute_task') and self._original_execute_task:
            self.orchestrator.execute_task = self._original_execute_task
    
    @asynccontextmanager
    async def hook_context(
        self,
        operation_name: str,
        agent_id: uuid.UUID,
        session_id: Optional[uuid.UUID] = None,
        parameters: Optional[Dict[str, Any]] = None,
        priority: int = 5
    ):
        """
        Context manager for automatic hook generation around operations.
        
        Usage:
            async with integration.hook_context("operation_name", agent_id) as ctx:
                result = await some_operation()
                ctx.set_result(result)
        """
        start_time = time.time()
        correlation_id = str(uuid.uuid4())
        
        class HookContext:
            def __init__(self):
                self.result = None
                self.error = None
            
            def set_result(self, result: Any) -> None:
                self.result = result
            
            def set_error(self, error: Exception) -> None:
                self.error = error
        
        context = HookContext()
        
        try:
            # Generate PreToolUse hook
            if self.hook_system:
                await self.hook_system.process_hook(
                    hook_type=HookType.PRE_TOOL_USE,
                    agent_id=agent_id,
                    session_id=session_id,
                    payload={
                        "tool_name": operation_name,
                        "parameters": parameters or {},
                        "operation_type": "context_managed"
                    },
                    correlation_id=correlation_id,
                    priority=priority
                )
            
            yield context
            
            # Generate successful PostToolUse hook
            execution_time = (time.time() - start_time) * 1000
            if self.hook_system:
                await self.hook_system.process_hook(
                    hook_type=HookType.POST_TOOL_USE,
                    agent_id=agent_id,
                    session_id=session_id,
                    payload={
                        "tool_name": operation_name,
                        "success": context.error is None,
                        "result": str(context.result)[:1000] if context.result else None,
                        "error": str(context.error) if context.error else None,
                        "execution_time_ms": execution_time,
                        "operation_type": "context_managed"
                    },
                    correlation_id=correlation_id,
                    priority=priority
                )
            
            self.metrics["hooks_generated"] += 2
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            # Generate error PostToolUse hook
            if self.hook_system:
                await self.hook_system.process_hook(
                    hook_type=HookType.POST_TOOL_USE,
                    agent_id=agent_id,
                    session_id=session_id,
                    payload={
                        "tool_name": operation_name,
                        "success": False,
                        "error": str(e),
                        "execution_time_ms": execution_time,
                        "operation_type": "context_managed"
                    },
                    correlation_id=correlation_id,
                    priority=1  # High priority for errors
                )
            
            self.metrics["hooks_failed"] += 1
            raise
    
    async def generate_performance_metrics_hook(
        self,
        agent_id: uuid.UUID,
        metrics_data: Dict[str, Any],
        session_id: Optional[uuid.UUID] = None
    ) -> HookProcessingResult:
        """Generate a performance metrics hook event."""
        if not self.config["track_performance_metrics"] or not self.hook_system:
            return HookProcessingResult(success=False, processing_time_ms=0.0, error="Performance tracking disabled")
        
        return await self.hook_system.process_notification(
            agent_id=agent_id,
            session_id=session_id,
            level="info",
            message="Performance metrics update",
            details={
                "metrics": metrics_data,
                "timestamp": datetime.utcnow().isoformat(),
                "metric_type": "performance"
            }
        )
    
    async def generate_inter_agent_communication_hook(
        self,
        source_agent_id: uuid.UUID,
        target_agent_id: uuid.UUID,
        message_type: str,
        message_data: Dict[str, Any],
        session_id: Optional[uuid.UUID] = None
    ) -> HookProcessingResult:
        """Generate an inter-agent communication hook event."""
        if not self.config["track_inter_agent_communication"] or not self.hook_system:
            return HookProcessingResult(success=False, processing_time_ms=0.0, error="Communication tracking disabled")
        
        return await self.hook_system.process_hook(
            hook_type=HookType.NOTIFICATION,
            agent_id=source_agent_id,
            session_id=session_id,
            payload={
                "level": "info",
                "message": f"Inter-agent communication: {message_type}",
                "details": {
                    "source_agent_id": str(source_agent_id),
                    "target_agent_id": str(target_agent_id),
                    "message_type": message_type,
                    "message_data": message_data,
                    "communication_timestamp": datetime.utcnow().isoformat()
                }
            },
            priority=4  # Medium priority for communication events
        )
    
    def enable_integration(self) -> None:
        """Enable hook integration."""
        self.integration_enabled = True
        logger.info("🔗 Hook integration enabled")
    
    def disable_integration(self) -> None:
        """Disable hook integration."""
        self.integration_enabled = False
        logger.info("🔗 Hook integration disabled")
    
    def configure(self, **config_updates) -> None:
        """Update integration configuration."""
        self.config.update(config_updates)
        logger.info(f"🔗 Hook integration configuration updated: {config_updates}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get integration metrics."""
        hook_system_metrics = {}
        if self.hook_system:
            hook_system_metrics = self.hook_system.get_comprehensive_metrics()
        
        return {
            "orchestrator_hook_integration": self.metrics.copy(),
            "hook_lifecycle_system": hook_system_metrics,
            "configuration": self.config.copy(),
            "integration_enabled": self.integration_enabled
        }


# Global orchestrator hook integration instance
_orchestrator_hook_integration: Optional[OrchestratorHookIntegration] = None


def get_orchestrator_hook_integration(orchestrator: AgentOrchestrator) -> OrchestratorHookIntegration:
    """Get or create the orchestrator hook integration instance."""
    global _orchestrator_hook_integration
    
    if _orchestrator_hook_integration is None:
        _orchestrator_hook_integration = OrchestratorHookIntegration(orchestrator)
    
    return _orchestrator_hook_integration


async def initialize_orchestrator_hooks(orchestrator: AgentOrchestrator) -> OrchestratorHookIntegration:
    """Initialize orchestrator hook integration."""
    integration = get_orchestrator_hook_integration(orchestrator)
    await integration.initialize()
    return integration


# Convenience decorators for hook integration

def with_hooks(
    operation_name: str,
    priority: int = 5,
    track_performance: bool = True
):
    """
    Decorator for automatic hook generation around operations.
    
    Usage:
        @with_hooks("my_operation", priority=3)
        async def my_operation(agent_id: uuid.UUID, session_id: Optional[uuid.UUID] = None):
            # operation logic
            return result
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract agent_id and session_id from arguments
            agent_id = kwargs.get('agent_id')
            session_id = kwargs.get('session_id')
            
            if not agent_id:
                # Try to extract from positional arguments
                if args and hasattr(args[0], 'id'):
                    agent_id = args[0].id
                else:
                    agent_id = uuid.uuid4()  # Fallback
            
            if isinstance(agent_id, str):
                agent_id = uuid.UUID(agent_id)
            
            # Get integration instance
            integration = _orchestrator_hook_integration
            if not integration:
                # No integration available, execute normally
                return await func(*args, **kwargs)
            
            # Use hook context
            async with integration.hook_context(
                operation_name=operation_name,
                agent_id=agent_id,
                session_id=session_id,
                parameters=kwargs,
                priority=priority
            ) as ctx:
                try:
                    result = await func(*args, **kwargs)
                    ctx.set_result(result)
                    return result
                except Exception as e:
                    ctx.set_error(e)
                    raise
        
        return wrapper
    return decorator


# Example usage functions for testing and demonstration
async def example_orchestrator_integration():
    """Example of how to use the orchestrator hook integration."""
    
    # Initialize orchestrator
    orchestrator = AgentOrchestrator()
    await orchestrator.start()
    
    try:
        # Initialize hook integration
        integration = await initialize_orchestrator_hooks(orchestrator)
        
        # Spawn an agent (hooks will be automatically generated)
        agent_id = await orchestrator.spawn_agent(AgentRole.STRATEGIC_PARTNER)
        
        # Generate custom performance metrics hook
        await integration.generate_performance_metrics_hook(
            agent_id=uuid.UUID(agent_id),
            metrics_data={
                "cpu_usage": 45.2,
                "memory_usage": 512.8,
                "response_time_ms": 234.5,
                "requests_per_second": 12.3
            }
        )
        
        # Generate inter-agent communication hook
        target_agent_id = await orchestrator.spawn_agent(AgentRole.PRODUCT_MANAGER)
        await integration.generate_inter_agent_communication_hook(
            source_agent_id=uuid.UUID(agent_id),
            target_agent_id=uuid.UUID(target_agent_id),
            message_type="task_delegation",
            message_data={
                "task_type": "analysis",
                "priority": "high",
                "deadline": "2024-01-15T10:00:00Z"
            }
        )
        
        # Get comprehensive metrics
        metrics = integration.get_metrics()
        logger.info(f"Integration metrics: {metrics}")
        
        # Shutdown agents (hooks will be automatically generated)
        await orchestrator.shutdown_agent(agent_id)
        await orchestrator.shutdown_agent(target_agent_id)
        
    finally:
        # Shutdown orchestrator
        await orchestrator.shutdown()
        
        # Shutdown integration
        if integration:
            await integration.shutdown()