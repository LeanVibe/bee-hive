"""
Hooks Integration Manager for LeanVibe Agent Hive 2.0 Observability

Seamlessly integrates Claude Code hook scripts with existing observability infrastructure
including database event storage, Redis streams, Prometheus metrics, and dashboard integration.
"""

import asyncio
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import structlog

# Import existing observability components
from app.observability.hooks.hooks_config import get_hook_config
# Define EventProcessor protocol locally to avoid circular imports
from typing import Protocol

class EventProcessor(Protocol):
    async def process_event(
        self,
        session_id: uuid.UUID,
        agent_id: uuid.UUID,
        event_type: 'EventType',
        payload: Dict[str, Any],
        latency_ms: Optional[int] = None
    ) -> str: ...

# Minimal HookInterceptor implementation for integration
class HookInterceptor:
    def __init__(self, event_processor: EventProcessor, enabled: bool = True, max_payload_size: int = 50000):
        self.event_processor = event_processor
        self._enabled = enabled
        self.max_payload_size = max_payload_size
    
    def enable(self) -> None:
        self._enabled = True
    
    def disable(self) -> None:
        self._enabled = False
from app.models.observability import AgentEvent, EventType
from app.core.database import get_db_session

logger = structlog.get_logger()


class HookEventProcessor:
    """
    Event processor that integrates with existing observability infrastructure.
    
    Implements the EventProcessor protocol to handle events from hook scripts
    and route them through the existing event processing pipeline.
    """
    
    def __init__(self):
        """Initialize hook event processor."""
        self.config = get_hook_config()
        
        logger.debug(
            "🔧 HookEventProcessor initialized",
            config_environment=self.config.environment
        )
    
    async def process_event(
        self,
        session_id: uuid.UUID,
        agent_id: uuid.UUID,
        event_type: EventType,
        payload: Dict[str, Any],
        latency_ms: Optional[int] = None
    ) -> str:
        """
        Process a captured event by storing it and routing through observability pipeline.
        
        Args:
            session_id: Session UUID
            agent_id: Agent UUID
            event_type: Type of event being processed
            payload: Event payload data
            latency_ms: Optional latency measurement
            
        Returns:
            Event ID as string
        """
        try:
            # Create AgentEvent instance
            event = AgentEvent(
                session_id=session_id,
                agent_id=agent_id,
                event_type=event_type,
                payload=payload,
                latency_ms=latency_ms
            )
            
            # Store in database if enabled
            event_id = None
            if self.config.integration.use_database:
                event_id = await self._store_event_in_database(event)
            
            # Publish to Redis streams if enabled
            if self.config.integration.use_redis_streams:
                await self._publish_to_redis_stream(event)
            
            # Update Prometheus metrics if enabled
            if self.config.integration.use_prometheus:
                await self._update_prometheus_metrics(event)
            
            # Integrate with existing observability middleware
            await self._integrate_with_observability_middleware(event)
            
            return str(event_id) if event_id else str(uuid.uuid4())
            
        except Exception as e:
            logger.error(
                "❌ Failed to process event",
                session_id=str(session_id),
                agent_id=str(agent_id),
                event_type=event_type.value,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def _store_event_in_database(self, event: AgentEvent) -> Optional[int]:
        """Store event in database using existing database infrastructure."""
        try:
            async with get_db_session() as session:
                session.add(event)
                await session.commit()
                await session.refresh(event)
                return event.id
        except Exception as e:
            logger.error(
                "❌ Failed to store event in database",
                event_type=event.event_type.value,
                error=str(e)
            )
            return None
    
    async def _publish_to_redis_stream(self, event: AgentEvent) -> None:
        """Publish event to Redis streams using existing Redis infrastructure."""
        try:
            from app.core.redis import get_redis_client
            
            redis = await get_redis_client()
            stream_data = {
                "event_type": event.event_type.value,
                "session_id": str(event.session_id),
                "agent_id": str(event.agent_id),
                "timestamp": datetime.utcnow().isoformat(),
                "payload": json.dumps(event.payload),
                "latency_ms": event.latency_ms
            }
            
            await redis.xadd(
                self.config.integration.redis_stream_key,
                stream_data,
                maxlen=self.config.integration.redis_max_len
            )
            
        except Exception as e:
            logger.error(
                "❌ Failed to publish event to Redis stream",
                event_type=event.event_type.value,
                error=str(e)
            )
    
    async def _update_prometheus_metrics(self, event: AgentEvent) -> None:
        """Update Prometheus metrics using existing metrics infrastructure."""
        try:
            # This would integrate with existing Prometheus setup
            from app.observability.prometheus_exporter import update_event_metrics
            
            await update_event_metrics(
                event_type=event.event_type.value,
                success=event.payload.get("success"),
                latency_ms=event.latency_ms,
                tool_name=event.payload.get("tool_name")
            )
            
        except ImportError:
            # Fallback if prometheus_exporter not available
            logger.debug(
                "📊 Prometheus metrics update (mock)",
                event_type=event.event_type.value
            )
        except Exception as e:
            logger.error(
                "❌ Failed to update Prometheus metrics",
                event_type=event.event_type.value,
                error=str(e)
            )
    
    async def _integrate_with_observability_middleware(self, event: AgentEvent) -> None:
        """Integrate with existing observability middleware."""
        try:
            # This would integrate with existing observability middleware
            from app.observability.middleware import process_agent_event
            
            await process_agent_event(event)
            
        except ImportError:
            # Fallback if middleware not available
            logger.debug(
                "🔌 Observability middleware integration (mock)",
                event_type=event.event_type.value
            )
        except Exception as e:
            logger.error(
                "❌ Failed to integrate with observability middleware",
                event_type=event.event_type.value,
                error=str(e)
            )


class HookScriptExecutor:
    """
    Executes Claude Code hook scripts and manages their lifecycle.
    
    Provides methods to execute hook scripts in various modes (subprocess, async)
    with proper error handling, timeout management, and result processing.
    """
    
    def __init__(self):
        """Initialize hook script executor."""
        self.config = get_hook_config()
        
        logger.debug(
            "🔧 HookScriptExecutor initialized",
            hooks_directory=str(self.config.hooks_directory)
        )
    
    async def execute_pre_tool_use_hook(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Execute pre-tool-use hook script.
        
        Args:
            tool_name: Name of the tool being executed
            parameters: Tool parameters
            session_id: Optional session ID
            agent_id: Optional agent ID
            correlation_id: Optional correlation ID
            
        Returns:
            Event ID if successful, None otherwise
        """
        if not self.config.enable_pre_tool_use:
            return None
        
        script_data = {
            "tool_name": tool_name,
            "parameters": parameters,
            "session_id": session_id,
            "agent_id": agent_id,
            "correlation_id": correlation_id
        }
        
        return await self._execute_hook_script(
            script_path=self.config.pre_tool_use_script,
            script_data=script_data,
            hook_type="pre_tool_use"
        )
    
    async def execute_post_tool_use_hook(
        self,
        tool_name: str,
        success: bool,
        execution_time_ms: Optional[int] = None,
        result: Any = None,
        error: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Execute post-tool-use hook script.
        
        Args:
            tool_name: Name of the tool that was executed
            success: Whether execution was successful
            execution_time_ms: Execution time in milliseconds
            result: Tool result data
            error: Error message if failed
            session_id: Optional session ID
            agent_id: Optional agent ID
            correlation_id: Optional correlation ID
            
        Returns:
            Event ID if successful, None otherwise
        """
        if not self.config.enable_post_tool_use:
            return None
        
        script_data = {
            "tool_name": tool_name,
            "success": success,
            "execution_time_ms": execution_time_ms,
            "result": result,
            "error": error,
            "session_id": session_id,
            "agent_id": agent_id,
            "correlation_id": correlation_id
        }
        
        return await self._execute_hook_script(
            script_path=self.config.post_tool_use_script,
            script_data=script_data,
            hook_type="post_tool_use"
        )
    
    async def execute_session_lifecycle_hook(
        self,
        event_type: str,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Execute session lifecycle hook script.
        
        Args:
            event_type: Type of session event (session_start, session_end, sleep, wake)
            session_id: Optional session ID
            agent_id: Optional agent ID
            **kwargs: Additional event-specific parameters
            
        Returns:
            Event ID if successful, None otherwise
        """
        if not self.config.enable_session_lifecycle:
            return None
        
        script_args = [event_type]
        
        # Add event-specific parameters
        if event_type in ("session_start", "wake") and kwargs.get("context_data"):
            script_args.append(json.dumps(kwargs["context_data"]))
        elif event_type in ("session_end", "sleep") and kwargs.get("reason"):
            script_args.append(kwargs["reason"])
        
        return await self._execute_hook_script_with_args(
            script_path=self.config.session_lifecycle_script,
            script_args=script_args,
            hook_type="session_lifecycle",
            env_vars={
                "CLAUDE_SESSION_ID": session_id,
                "CLAUDE_AGENT_ID": agent_id
            }
        )
    
    async def _execute_hook_script(
        self,
        script_path: Path,
        script_data: Dict[str, Any],
        hook_type: str
    ) -> Optional[str]:
        """
        Execute a hook script with JSON data via stdin.
        
        Args:
            script_path: Path to the hook script
            script_data: Data to pass to the script
            hook_type: Type of hook for logging
            
        Returns:
            Event ID if successful, None otherwise
        """
        if not script_path.exists():
            logger.error(
                f"❌ Hook script not found: {script_path}",
                hook_type=hook_type
            )
            return None
        
        try:
            # Prepare environment variables
            env = os.environ.copy()
            if script_data.get("session_id"):
                env["CLAUDE_SESSION_ID"] = script_data["session_id"]
            if script_data.get("agent_id"):
                env["CLAUDE_AGENT_ID"] = script_data["agent_id"]
            
            # Execute script with JSON data via stdin
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(script_path),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            # Send JSON data to stdin
            input_data = json.dumps(script_data).encode()
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input_data),
                timeout=30.0  # 30 second timeout
            )
            
            # Check result
            if process.returncode == 0:
                result = stdout.decode().strip()
                logger.debug(
                    f"✅ {hook_type} hook executed successfully",
                    script_path=str(script_path),
                    result=result
                )
                return result
            else:
                error_msg = stderr.decode().strip()
                logger.error(
                    f"❌ {hook_type} hook failed",
                    script_path=str(script_path),
                    return_code=process.returncode,
                    error=error_msg
                )
                return None
                
        except asyncio.TimeoutError:
            logger.error(
                f"⏰ {hook_type} hook timed out",
                script_path=str(script_path),
                timeout_seconds=30
            )
            return None
        except Exception as e:
            logger.error(
                f"💥 {hook_type} hook execution failed",
                script_path=str(script_path),
                error=str(e),
                exc_info=True
            )
            return None
    
    async def _execute_hook_script_with_args(
        self,
        script_path: Path,
        script_args: List[str],
        hook_type: str,
        env_vars: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """
        Execute a hook script with command line arguments.
        
        Args:
            script_path: Path to the hook script
            script_args: Command line arguments for the script
            hook_type: Type of hook for logging
            env_vars: Optional environment variables
            
        Returns:
            Event ID if successful, None otherwise
        """
        if not script_path.exists():
            logger.error(
                f"❌ Hook script not found: {script_path}",
                hook_type=hook_type
            )
            return None
        
        try:
            # Prepare environment variables
            env = os.environ.copy()
            if env_vars:
                env.update({k: v for k, v in env_vars.items() if v})
            
            # Build command
            cmd = [sys.executable, str(script_path)] + script_args
            
            # Execute script with arguments
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=30.0  # 30 second timeout
            )
            
            # Check result
            if process.returncode == 0:
                result = stdout.decode().strip()
                logger.debug(
                    f"✅ {hook_type} hook executed successfully",
                    script_path=str(script_path),
                    args=script_args,
                    result=result
                )
                return result
            else:
                error_msg = stderr.decode().strip()
                logger.error(
                    f"❌ {hook_type} hook failed",
                    script_path=str(script_path),
                    args=script_args,
                    return_code=process.returncode,
                    error=error_msg
                )
                return None
                
        except asyncio.TimeoutError:
            logger.error(
                f"⏰ {hook_type} hook timed out",
                script_path=str(script_path),
                args=script_args,
                timeout_seconds=30
            )
            return None
        except Exception as e:
            logger.error(
                f"💥 {hook_type} hook execution failed",
                script_path=str(script_path),
                args=script_args,
                error=str(e),
                exc_info=True
            )
            return None


class HookIntegrationManager:
    """
    Main integration manager that coordinates hook scripts with observability infrastructure.
    
    Provides a unified interface for executing hook scripts and ensuring events
    are properly processed through the existing observability pipeline.
    """
    
    def __init__(self):
        """Initialize hook integration manager."""
        self.config = get_hook_config()
        self.event_processor = HookEventProcessor()
        self.script_executor = HookScriptExecutor()
        
        # Initialize hook interceptor with our event processor
        self.hook_interceptor = HookInterceptor(
            event_processor=self.event_processor,
            enabled=True,
            max_payload_size=self.config.security.max_payload_size
        )
        
        logger.info(
            "🔗 HookIntegrationManager initialized",
            environment=self.config.environment,
            hooks_enabled=self._get_enabled_hooks()
        )
    
    def _get_enabled_hooks(self) -> Dict[str, bool]:
        """Get status of enabled hooks."""
        return {
            "pre_tool_use": self.config.enable_pre_tool_use,
            "post_tool_use": self.config.enable_post_tool_use,
            "session_lifecycle": self.config.enable_session_lifecycle,
            "error_hooks": self.config.enable_error_hooks
        }
    
    async def capture_tool_execution(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        execution_result: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> Dict[str, Optional[str]]:
        """
        Capture complete tool execution lifecycle (pre + post).
        
        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            execution_result: Optional execution result data
            session_id: Optional session ID
            agent_id: Optional agent ID
            
        Returns:
            Dictionary with pre_event_id and post_event_id
        """
        results = {"pre_event_id": None, "post_event_id": None}
        
        # Generate correlation ID for linking events
        correlation_id = str(uuid.uuid4())
        
        # Execute pre-tool-use hook
        if self.config.enable_pre_tool_use:
            results["pre_event_id"] = await self.script_executor.execute_pre_tool_use_hook(
                tool_name=tool_name,
                parameters=parameters,
                session_id=session_id,
                agent_id=agent_id,
                correlation_id=correlation_id
            )
        
        # Execute post-tool-use hook if result provided
        if execution_result and self.config.enable_post_tool_use:
            results["post_event_id"] = await self.script_executor.execute_post_tool_use_hook(
                tool_name=tool_name,
                success=execution_result.get("success", True),
                execution_time_ms=execution_result.get("execution_time_ms"),
                result=execution_result.get("result"),
                error=execution_result.get("error"),
                session_id=session_id,
                agent_id=agent_id,
                correlation_id=correlation_id
            )
        
        return results
    
    async def capture_session_lifecycle_event(
        self,
        event_type: str,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Capture session lifecycle event.
        
        Args:
            event_type: Type of event (session_start, session_end, sleep, wake)
            session_id: Optional session ID
            agent_id: Optional agent ID
            **kwargs: Additional event data
            
        Returns:
            Event ID if successful, None otherwise
        """
        return await self.script_executor.execute_session_lifecycle_hook(
            event_type=event_type,
            session_id=session_id,
            agent_id=agent_id,
            **kwargs
        )
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """
        Get comprehensive integration status.
        
        Returns:
            Status dictionary with component health and configuration
        """
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "environment": self.config.environment,
            "hooks_enabled": self._get_enabled_hooks(),
            "script_validation": {},
            "integration_health": {},
            "configuration": {
                "database_enabled": self.config.integration.use_database,
                "redis_enabled": self.config.integration.use_redis_streams,
                "prometheus_enabled": self.config.integration.use_prometheus,
                "webhooks_configured": len(self.config.integration.webhook_urls) > 0
            }
        }
        
        # Validate script existence
        scripts = {
            "pre_tool_use": self.config.pre_tool_use_script,
            "post_tool_use": self.config.post_tool_use_script,
            "session_lifecycle": self.config.session_lifecycle_script
        }
        
        for script_name, script_path in scripts.items():
            status["script_validation"][script_name] = {
                "exists": script_path.exists(),
                "executable": os.access(script_path, os.X_OK) if script_path.exists() else False,
                "path": str(script_path)
            }
        
        # Test integration components
        try:
            # Test database connection
            async with get_db_session() as session:
                status["integration_health"]["database"] = "healthy"
        except Exception as e:
            status["integration_health"]["database"] = f"error: {str(e)}"
        
        try:
            # Test Redis connection
            from app.core.redis import get_redis_client
            redis = await get_redis_client()
            await redis.ping()
            status["integration_health"]["redis"] = "healthy"
        except Exception as e:
            status["integration_health"]["redis"] = f"error: {str(e)}"
        
        return status
    
    def enable_hooks(self, hook_types: Optional[List[str]] = None) -> None:
        """
        Enable specific hooks or all hooks.
        
        Args:
            hook_types: List of hook types to enable, or None for all
        """
        if hook_types is None:
            hook_types = ["pre_tool_use", "post_tool_use", "session_lifecycle", "error_hooks"]
        
        for hook_type in hook_types:
            if hook_type == "pre_tool_use":
                self.config.enable_pre_tool_use = True
            elif hook_type == "post_tool_use":
                self.config.enable_post_tool_use = True
            elif hook_type == "session_lifecycle":
                self.config.enable_session_lifecycle = True
            elif hook_type == "error_hooks":
                self.config.enable_error_hooks = True
        
        self.hook_interceptor.enable()
        
        logger.info(
            "✅ Hooks enabled",
            enabled_hooks=hook_types
        )
    
    def disable_hooks(self, hook_types: Optional[List[str]] = None) -> None:
        """
        Disable specific hooks or all hooks.
        
        Args:
            hook_types: List of hook types to disable, or None for all
        """
        if hook_types is None:
            hook_types = ["pre_tool_use", "post_tool_use", "session_lifecycle", "error_hooks"]
            self.hook_interceptor.disable()
        
        for hook_type in hook_types:
            if hook_type == "pre_tool_use":
                self.config.enable_pre_tool_use = False
            elif hook_type == "post_tool_use":
                self.config.enable_post_tool_use = False
            elif hook_type == "session_lifecycle":
                self.config.enable_session_lifecycle = False
            elif hook_type == "error_hooks":
                self.config.enable_error_hooks = False
        
        logger.info(
            "❌ Hooks disabled",
            disabled_hooks=hook_types
        )


# Global integration manager instance
_integration_manager: Optional[HookIntegrationManager] = None


def get_hook_integration_manager() -> HookIntegrationManager:
    """Get the global hook integration manager instance."""
    global _integration_manager
    if _integration_manager is None:
        _integration_manager = HookIntegrationManager()
    return _integration_manager


def set_hook_integration_manager(manager: HookIntegrationManager) -> None:
    """Set the global hook integration manager instance."""
    global _integration_manager
    _integration_manager = manager
    logger.info("🔗 Global hook integration manager set")


def clear_hook_integration_manager() -> None:
    """Clear the global hook integration manager instance."""
    global _integration_manager
    _integration_manager = None
    logger.info("🔗 Global hook integration manager cleared")