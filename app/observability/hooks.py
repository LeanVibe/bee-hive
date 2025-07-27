"""
Hook Interceptor for LeanVibe Agent Hive 2.0 Observability System

Captures agent lifecycle events including PreToolUse, PostToolUse, Notification,
Stop, and SubagentStop events. Provides comprehensive event tracking for debugging,
monitoring, and performance optimization.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Protocol

import structlog

from app.models.observability import EventType

logger = structlog.get_logger()


class EventProcessor(Protocol):
    """Protocol for event processors that handle captured events."""
    
    async def process_event(
        self,
        session_id: uuid.UUID,
        agent_id: uuid.UUID,
        event_type: EventType,
        payload: Dict[str, Any],
        latency_ms: Optional[int] = None
    ) -> str:
        """Process a captured event."""
        ...


class EventCapture:
    """
    Utility class for standardizing event data capture.
    
    Provides consistent event payload creation and validation across
    different event types.
    """
    
    @staticmethod
    def create_pre_tool_use_payload(
        tool_name: str,
        parameters: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create standardized PreToolUse event payload."""
        payload = {
            "tool_name": tool_name,
            "parameters": parameters,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if correlation_id:
            payload["correlation_id"] = correlation_id
            
        return payload
    
    @staticmethod
    def create_post_tool_use_payload(
        tool_name: str,
        success: bool,
        result: Any = None,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
        execution_time_ms: Optional[int] = None,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create standardized PostToolUse event payload."""
        payload = {
            "tool_name": tool_name,
            "success": success,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if result is not None:
            # Handle large results to prevent payload bloat
            if isinstance(result, str) and len(result) > 10000:
                payload["result"] = result[:10000] + "... (truncated)"
                payload["result_truncated"] = True
                payload["full_result_size"] = len(result)
            else:
                payload["result"] = result
        
        if error:
            payload["error"] = error
            
        if error_type:
            payload["error_type"] = error_type
            
        if execution_time_ms is not None:
            payload["execution_time_ms"] = execution_time_ms
            
        if correlation_id:
            payload["correlation_id"] = correlation_id
            
        return payload
    
    @staticmethod
    def create_notification_payload(
        level: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create standardized Notification event payload."""
        payload = {
            "level": level,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if details:
            payload["details"] = details
            
        return payload
    
    @staticmethod
    def create_stop_payload(
        reason: str,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create standardized Stop event payload."""
        payload = {
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if details:
            payload["details"] = details
            
        return payload
    
    @staticmethod
    def create_subagent_stop_payload(
        subagent_id: uuid.UUID,
        reason: str,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create standardized SubagentStop event payload."""
        payload = {
            "subagent_id": str(subagent_id),
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if details:
            payload["details"] = details
            
        return payload


class HookInterceptor:
    """
    Hook Interceptor for capturing agent lifecycle events.
    
    Integrates with Claude Code hooks to capture PreToolUse, PostToolUse,
    and other critical events for observability and monitoring.
    """
    
    def __init__(
        self,
        event_processor: EventProcessor,
        enabled: bool = True,
        max_payload_size: int = 50000
    ):
        """
        Initialize HookInterceptor.
        
        Args:
            event_processor: Event processor for handling captured events
            enabled: Whether event capture is enabled
            max_payload_size: Maximum size for event payloads (bytes)
        """
        self.event_processor = event_processor
        self._enabled = enabled
        self.max_payload_size = max_payload_size
        
        logger.info(
            "🔌 HookInterceptor initialized",
            enabled=enabled,
            max_payload_size=max_payload_size
        )
    
    @property
    def is_enabled(self) -> bool:
        """Check if event capture is enabled."""
        return self._enabled
    
    def enable(self) -> None:
        """Enable event capture."""
        self._enabled = True
        logger.info("✅ HookInterceptor enabled")
    
    def disable(self) -> None:
        """Disable event capture."""
        self._enabled = False
        logger.info("❌ HookInterceptor disabled")
    
    async def capture_pre_tool_use(
        self,
        session_id: uuid.UUID,
        agent_id: uuid.UUID,
        tool_data: Dict[str, Any]
    ) -> Optional[str]:
        """
        Capture PreToolUse event.
        
        Args:
            session_id: Session UUID
            agent_id: Agent UUID
            tool_data: Tool execution data including name and parameters
            
        Returns:
            Event ID if processed, None if disabled
        """
        if not self._enabled:
            return None
            
        try:
            # Extract tool information
            tool_name = tool_data.get("tool_name", "unknown")
            parameters = tool_data.get("parameters", {})
            correlation_id = tool_data.get("correlation_id")
            
            # Create standardized payload
            payload = EventCapture.create_pre_tool_use_payload(
                tool_name=tool_name,
                parameters=parameters,
                correlation_id=correlation_id
            )
            
            # Process event
            event_id = await self.event_processor.process_event(
                session_id=session_id,
                agent_id=agent_id,
                event_type=EventType.PRE_TOOL_USE,
                payload=payload
            )
            
            logger.debug(
                "📝 PreToolUse event captured",
                event_id=event_id,
                tool_name=tool_name,
                session_id=str(session_id),
                agent_id=str(agent_id)
            )
            
            return event_id
            
        except Exception as e:
            logger.error(
                "❌ Failed to capture PreToolUse event",
                error=str(e),
                session_id=str(session_id),
                agent_id=str(agent_id),
                exc_info=True
            )
            return None
    
    async def capture_post_tool_use(
        self,
        session_id: uuid.UUID,
        agent_id: uuid.UUID,
        tool_result: Dict[str, Any],
        latency_ms: Optional[int] = None
    ) -> Optional[str]:
        """
        Capture PostToolUse event.
        
        Args:
            session_id: Session UUID
            agent_id: Agent UUID
            tool_result: Tool execution result including success/failure info
            latency_ms: Optional latency measurement
            
        Returns:
            Event ID if processed, None if disabled
        """
        if not self._enabled:
            return None
            
        try:
            # Extract tool result information
            tool_name = tool_result.get("tool_name", "unknown")
            success = tool_result.get("success", True)
            result = tool_result.get("result")
            error = tool_result.get("error")
            error_type = tool_result.get("error_type")
            execution_time_ms = tool_result.get("execution_time_ms")
            correlation_id = tool_result.get("correlation_id")
            
            # Create standardized payload
            payload = EventCapture.create_post_tool_use_payload(
                tool_name=tool_name,
                success=success,
                result=result,
                error=error,
                error_type=error_type,
                execution_time_ms=execution_time_ms,
                correlation_id=correlation_id
            )
            
            # Process event
            event_id = await self.event_processor.process_event(
                session_id=session_id,
                agent_id=agent_id,
                event_type=EventType.POST_TOOL_USE,
                payload=payload,
                latency_ms=latency_ms
            )
            
            logger.debug(
                "📝 PostToolUse event captured",
                event_id=event_id,
                tool_name=tool_name,
                success=success,
                latency_ms=latency_ms,
                session_id=str(session_id),
                agent_id=str(agent_id)
            )
            
            return event_id
            
        except Exception as e:
            logger.error(
                "❌ Failed to capture PostToolUse event",
                error=str(e),
                session_id=str(session_id),
                agent_id=str(agent_id),
                exc_info=True
            )
            return None
    
    async def capture_notification(
        self,
        session_id: uuid.UUID,
        agent_id: uuid.UUID,
        notification: Dict[str, Any]
    ) -> Optional[str]:
        """
        Capture Notification event.
        
        Args:
            session_id: Session UUID
            agent_id: Agent UUID
            notification: Notification data including level and message
            
        Returns:
            Event ID if processed, None if disabled
        """
        if not self._enabled:
            return None
            
        try:
            # Extract notification information
            level = notification.get("level", "info")
            message = notification.get("message", "")
            details = notification.get("details")
            
            # Create standardized payload
            payload = EventCapture.create_notification_payload(
                level=level,
                message=message,
                details=details
            )
            
            # Process event
            event_id = await self.event_processor.process_event(
                session_id=session_id,
                agent_id=agent_id,
                event_type=EventType.NOTIFICATION,
                payload=payload
            )
            
            logger.debug(
                "📝 Notification event captured",
                event_id=event_id,
                level=level,
                message=message,
                session_id=str(session_id),
                agent_id=str(agent_id)
            )
            
            return event_id
            
        except Exception as e:
            logger.error(
                "❌ Failed to capture Notification event",
                error=str(e),
                session_id=str(session_id),
                agent_id=str(agent_id),
                exc_info=True
            )
            return None
    
    async def capture_stop(
        self,
        session_id: uuid.UUID,
        agent_id: uuid.UUID,
        reason: str,
        details: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Capture Stop event.
        
        Args:
            session_id: Session UUID
            agent_id: Agent UUID
            reason: Reason for stopping
            details: Optional additional details
            
        Returns:
            Event ID if processed, None if disabled
        """
        if not self._enabled:
            return None
            
        try:
            # Create standardized payload
            payload = EventCapture.create_stop_payload(
                reason=reason,
                details=details
            )
            
            # Process event
            event_id = await self.event_processor.process_event(
                session_id=session_id,
                agent_id=agent_id,
                event_type=EventType.STOP,
                payload=payload
            )
            
            logger.info(
                "🛑 Stop event captured",
                event_id=event_id,
                reason=reason,
                session_id=str(session_id),
                agent_id=str(agent_id)
            )
            
            return event_id
            
        except Exception as e:
            logger.error(
                "❌ Failed to capture Stop event",
                error=str(e),
                session_id=str(session_id),
                agent_id=str(agent_id),
                exc_info=True
            )
            return None
    
    async def capture_subagent_stop(
        self,
        session_id: uuid.UUID,
        agent_id: uuid.UUID,
        subagent_id: uuid.UUID,
        reason: str,
        details: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Capture SubagentStop event.
        
        Args:
            session_id: Session UUID
            agent_id: Parent agent UUID
            subagent_id: Subagent UUID that stopped
            reason: Reason for stopping
            details: Optional additional details
            
        Returns:
            Event ID if processed, None if disabled
        """
        if not self._enabled:
            return None
            
        try:
            # Create standardized payload
            payload = EventCapture.create_subagent_stop_payload(
                subagent_id=subagent_id,
                reason=reason,
                details=details
            )
            
            # Process event
            event_id = await self.event_processor.process_event(
                session_id=session_id,
                agent_id=agent_id,
                event_type=EventType.SUBAGENT_STOP,
                payload=payload
            )
            
            logger.info(
                "🛑 SubagentStop event captured",
                event_id=event_id,
                subagent_id=str(subagent_id),
                reason=reason,
                session_id=str(session_id),
                agent_id=str(agent_id)
            )
            
            return event_id
            
        except Exception as e:
            logger.error(
                "❌ Failed to capture SubagentStop event",
                error=str(e),
                session_id=str(session_id),
                agent_id=str(agent_id),
                subagent_id=str(subagent_id),
                exc_info=True
            )
            return None
    
    async def capture_batch(
        self,
        events: list[Dict[str, Any]]
    ) -> list[Optional[str]]:
        """
        Capture multiple events in batch for high-throughput scenarios.
        
        Args:
            events: List of event dictionaries containing all event data
            
        Returns:
            List of event IDs (None for failed captures)
        """
        if not self._enabled:
            return [None] * len(events)
            
        tasks = []
        for event in events:
            event_type = event.get("event_type")
            session_id = event.get("session_id")
            agent_id = event.get("agent_id")
            
            if event_type == "PreToolUse":
                task = self.capture_pre_tool_use(
                    session_id=session_id,
                    agent_id=agent_id,
                    tool_data=event.get("tool_data", {})
                )
            elif event_type == "PostToolUse":
                task = self.capture_post_tool_use(
                    session_id=session_id,
                    agent_id=agent_id,
                    tool_result=event.get("tool_result", {}),
                    latency_ms=event.get("latency_ms")
                )
            elif event_type == "Notification":
                task = self.capture_notification(
                    session_id=session_id,
                    agent_id=agent_id,
                    notification=event.get("notification", {})
                )
            else:
                # Unsupported event type
                tasks.append(asyncio.create_task(asyncio.sleep(0, result=None)))
                continue
                
            tasks.append(asyncio.create_task(task))
        
        # Execute all captures concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to None
        event_ids = []
        for result in results:
            if isinstance(result, Exception):
                logger.error("❌ Batch event capture failed", error=str(result))
                event_ids.append(None)
            else:
                event_ids.append(result)
        
        logger.debug(
            "📦 Batch event capture completed",
            total_events=len(events),
            successful=sum(1 for eid in event_ids if eid is not None),
            failed=sum(1 for eid in event_ids if eid is None)
        )
        
        return event_ids


# Global hook interceptor instance
_hook_interceptor: Optional[HookInterceptor] = None


def get_hook_interceptor() -> Optional[HookInterceptor]:
    """Get the global hook interceptor instance."""
    return _hook_interceptor


def set_hook_interceptor(interceptor: HookInterceptor) -> None:
    """Set the global hook interceptor instance."""
    global _hook_interceptor
    _hook_interceptor = interceptor
    logger.info("🔗 Global hook interceptor set")


def clear_hook_interceptor() -> None:
    """Clear the global hook interceptor instance."""
    global _hook_interceptor
    _hook_interceptor = None
    logger.info("🔗 Global hook interceptor cleared")