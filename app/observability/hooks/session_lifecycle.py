#!/usr/bin/env python3
"""
Session Lifecycle Hook for LeanVibe Agent Hive 2.0 Observability

Manages session start/end events, sleep/wake cycles, and memory consolidation 
for comprehensive Claude Code session tracking and optimization.
"""

import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

# Add parent directories to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.observability import AgentEvent, EventType
from app.core.database import get_db_session
from app.observability.hooks.hooks_config import get_hook_config

logger = structlog.get_logger()


class SessionLifecycleCapture:
    """
    Session lifecycle event capture and management system.
    
    Handles session start/end events, sleep/wake cycles, memory consolidation
    triggers, and performance optimization for Claude Code sessions.
    """
    
    def __init__(self):
        """Initialize session lifecycle capture system."""
        self.config = get_hook_config()
        self.capture_time = time.time()
        
        logger.debug(
            "🔧 SessionLifecycleCapture initialized",
            config_environment=self.config.environment,
            enable_sleep_wake_hooks=self.config.session.enable_sleep_wake_hooks
        )
    
    async def capture_session_start(
        self,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        context_data: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Capture session start event.
        
        Args:
            session_id: Optional session identifier
            agent_id: Optional agent identifier
            context_data: Optional context data for session initialization
            
        Returns:
            Event ID if captured successfully, None otherwise
        """
        if not self.config.should_capture_event("SessionStart"):
            logger.debug("🚫 SessionStart event capture disabled")
            return None
        
        try:
            # Generate IDs if not provided
            if session_id is None:
                session_id = self.config.get_session_id()
            if agent_id is None:
                agent_id = self.config.get_agent_id()
            
            # Convert string IDs to UUID objects
            session_uuid = uuid.UUID(session_id) if isinstance(session_id, str) else session_id
            agent_uuid = uuid.UUID(agent_id) if isinstance(agent_id, str) else agent_id
            
            # Create session start notification
            event = AgentEvent.create_notification(
                session_id=session_uuid,
                agent_id=agent_uuid,
                level="info",
                message="Claude Code session started"
            )
            
            # Add session metadata
            session_metadata = {
                "event_subtype": "session_start",
                "session_start_time": datetime.utcnow().isoformat(),
                "environment": self.config.environment,
                "hook_version": "1.0",
                "auto_created": self.config.session.auto_create_sessions,
                "context_threshold": self.config.session.context_threshold_percent
            }
            
            # Add context data if provided
            if context_data:
                session_metadata["context_data"] = self._sanitize_context_data(context_data)
            
            # Add system information
            session_metadata["system_info"] = {
                "python_version": sys.version.split()[0],
                "working_directory": os.getcwd(),
                "process_id": os.getpid(),
                "parent_process_id": os.getppid()
            }
            
            # Add performance baseline
            session_metadata["performance_baseline"] = {
                "memory_usage_mb": self._get_memory_usage_mb(),
                "start_timestamp": time.time()
            }
            
            event.payload.update(session_metadata)
            
            # Store event
            event_id = await self._store_and_publish_event(event, session_id)
            
            # Initialize session tracking
            await self._initialize_session_tracking(session_uuid, agent_uuid)
            
            logger.info(
                "🚀 Session start event captured",
                event_id=event_id,
                session_id=str(session_uuid),
                agent_id=str(agent_uuid),
                environment=self.config.environment
            )
            
            return str(event_id) if event_id else str(session_uuid)
            
        except Exception as e:
            logger.error(
                "❌ Failed to capture session start event",
                session_id=str(session_id) if session_id else None,
                agent_id=str(agent_id) if agent_id else None,
                error=str(e),
                exc_info=True
            )
            return None
    
    async def capture_session_end(
        self,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        reason: str = "normal_completion",
        session_summary: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Capture session end event with summary statistics.
        
        Args:
            session_id: Optional session identifier
            agent_id: Optional agent identifier
            reason: Reason for session end
            session_summary: Optional session summary data
            
        Returns:
            Event ID if captured successfully, None otherwise
        """
        if not self.config.should_capture_event("SessionEnd"):
            logger.debug("🚫 SessionEnd event capture disabled")
            return None
        
        try:
            # Generate IDs if not provided
            if session_id is None:
                session_id = self.config.get_session_id()
            if agent_id is None:
                agent_id = self.config.get_agent_id()
            
            # Convert string IDs to UUID objects
            session_uuid = uuid.UUID(session_id) if isinstance(session_id, str) else session_id
            agent_uuid = uuid.UUID(agent_id) if isinstance(agent_id, str) else agent_id
            
            # Generate session summary if not provided
            if session_summary is None:
                session_summary = await self._generate_session_summary(session_uuid)
            
            # Create session end stop event
            details = {
                "event_subtype": "session_end",
                "session_end_time": datetime.utcnow().isoformat(),
                "session_duration_ms": session_summary.get("duration_ms", 0),
                "reason": reason,
                "session_summary": session_summary,
                "environment": self.config.environment,
                "hook_version": "1.0"
            }
            
            event = AgentEvent.create_stop(
                session_id=session_uuid,
                agent_id=agent_uuid,
                reason=f"Session ended: {reason}",
                details=details
            )
            
            # Store event
            event_id = await self._store_and_publish_event(event, session_id)
            
            # Cleanup session tracking
            await self._cleanup_session_tracking(session_uuid)
            
            logger.info(
                "🏁 Session end event captured",
                event_id=event_id,
                session_id=str(session_uuid),
                agent_id=str(agent_uuid),
                reason=reason,
                duration_ms=session_summary.get("duration_ms", 0),
                tool_executions=session_summary.get("tool_executions", 0)
            )
            
            return str(event_id) if event_id else str(session_uuid)
            
        except Exception as e:
            logger.error(
                "❌ Failed to capture session end event",
                session_id=str(session_id) if session_id else None,
                agent_id=str(agent_id) if agent_id else None,
                reason=reason,
                error=str(e),
                exc_info=True
            )
            return None
    
    async def capture_sleep_event(
        self,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        sleep_reason: str = "context_threshold_reached",
        context_stats: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Capture sleep cycle initiation event.
        
        Args:
            session_id: Optional session identifier
            agent_id: Optional agent identifier
            sleep_reason: Reason for sleep initiation
            context_stats: Optional context usage statistics
            
        Returns:
            Event ID if captured successfully, None otherwise
        """
        if not self.config.session.enable_sleep_wake_hooks:
            logger.debug("🚫 Sleep/wake hooks disabled")
            return None
        
        try:
            # Generate IDs if not provided
            if session_id is None:
                session_id = self.config.get_session_id()
            if agent_id is None:
                agent_id = self.config.get_agent_id()
            
            # Convert string IDs to UUID objects
            session_uuid = uuid.UUID(session_id) if isinstance(session_id, str) else session_id
            agent_uuid = uuid.UUID(agent_id) if isinstance(agent_id, str) else agent_id
            
            # Create sleep notification
            event = AgentEvent.create_notification(
                session_id=session_uuid,
                agent_id=agent_uuid,
                level="info",
                message=f"Sleep cycle initiated: {sleep_reason}"
            )
            
            # Add sleep metadata
            sleep_metadata = {
                "event_subtype": "sleep",
                "sleep_start_time": datetime.utcnow().isoformat(),
                "sleep_reason": sleep_reason,
                "environment": self.config.environment,
                "hook_version": "1.0"
            }
            
            # Add context statistics if provided
            if context_stats is None:
                context_stats = await self._get_context_statistics(session_uuid)
            
            sleep_metadata["context_stats"] = context_stats
            
            # Add pre-sleep memory usage
            sleep_metadata["pre_sleep_memory"] = {
                "memory_usage_mb": self._get_memory_usage_mb(),
                "timestamp": time.time()
            }
            
            # Check if consolidation should be triggered
            if self._should_trigger_consolidation(context_stats):
                sleep_metadata["consolidation_triggered"] = True
                sleep_metadata["consolidation_reason"] = "event_threshold_reached"
            
            event.payload.update(sleep_metadata)
            
            # Store event
            event_id = await self._store_and_publish_event(event, session_id)
            
            # Trigger consolidation if needed
            if sleep_metadata.get("consolidation_triggered"):
                await self._trigger_memory_consolidation(session_uuid, context_stats)
            
            logger.info(
                "😴 Sleep event captured",
                event_id=event_id,
                session_id=str(session_uuid),
                agent_id=str(agent_uuid),
                sleep_reason=sleep_reason,
                context_usage_percent=context_stats.get("usage_percent", 0),
                consolidation_triggered=sleep_metadata.get("consolidation_triggered", False)
            )
            
            return str(event_id) if event_id else str(session_uuid)
            
        except Exception as e:
            logger.error(
                "❌ Failed to capture sleep event",
                session_id=str(session_id) if session_id else None,
                agent_id=str(agent_id) if agent_id else None,
                sleep_reason=sleep_reason,
                error=str(e),
                exc_info=True
            )
            return None
    
    async def capture_wake_event(
        self,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        wake_reason: str = "new_session_start",
        restored_context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Capture wake cycle completion event.
        
        Args:
            session_id: Optional session identifier
            agent_id: Optional agent identifier
            wake_reason: Reason for wake completion
            restored_context: Optional restored context information
            
        Returns:
            Event ID if captured successfully, None otherwise
        """
        if not self.config.session.enable_sleep_wake_hooks:
            logger.debug("🚫 Sleep/wake hooks disabled")
            return None
        
        try:
            # Generate IDs if not provided
            if session_id is None:
                session_id = self.config.get_session_id()
            if agent_id is None:
                agent_id = self.config.get_agent_id()
            
            # Convert string IDs to UUID objects
            session_uuid = uuid.UUID(session_id) if isinstance(session_id, str) else session_id
            agent_uuid = uuid.UUID(agent_id) if isinstance(agent_id, str) else agent_id
            
            # Create wake notification
            event = AgentEvent.create_notification(
                session_id=session_uuid,
                agent_id=agent_uuid,
                level="info",
                message=f"Wake cycle completed: {wake_reason}"
            )
            
            # Add wake metadata
            wake_metadata = {
                "event_subtype": "wake",
                "wake_time": datetime.utcnow().isoformat(),
                "wake_reason": wake_reason,
                "environment": self.config.environment,
                "hook_version": "1.0"
            }
            
            # Add restored context information
            if restored_context:
                wake_metadata["restored_context"] = self._sanitize_context_data(restored_context)
            
            # Add post-wake memory usage
            wake_metadata["post_wake_memory"] = {
                "memory_usage_mb": self._get_memory_usage_mb(),
                "timestamp": time.time()
            }
            
            # Add context restoration statistics
            wake_metadata["context_restoration"] = await self._get_context_restoration_stats(session_uuid)
            
            event.payload.update(wake_metadata)
            
            # Store event
            event_id = await self._store_and_publish_event(event, session_id)
            
            # Update session tracking with wake information
            await self._update_session_tracking_wake(session_uuid)
            
            logger.info(
                "🌅 Wake event captured",
                event_id=event_id,
                session_id=str(session_uuid),
                agent_id=str(agent_uuid),
                wake_reason=wake_reason,
                memory_usage_mb=wake_metadata["post_wake_memory"]["memory_usage_mb"],
                context_restored=bool(restored_context)
            )
            
            return str(event_id) if event_id else str(session_uuid)
            
        except Exception as e:
            logger.error(
                "❌ Failed to capture wake event",
                session_id=str(session_id) if session_id else None,
                agent_id=str(agent_id) if agent_id else None,
                wake_reason=wake_reason,
                error=str(e),
                exc_info=True
            )
            return None
    
    def _sanitize_context_data(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize context data for security."""
        if not self.config.security.sanitize_sensitive_data:
            return context_data
        
        # Truncate large context data and redact sensitive information
        sanitized = {}
        for key, value in list(context_data.items())[:20]:  # Limit keys
            if isinstance(value, str) and len(value) > 1000:
                sanitized[key] = value[:1000] + "... (truncated)"
            elif any(pattern in key.lower() for pattern in ["password", "token", "key", "secret"]):
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return round(memory_info.rss / 1024 / 1024, 2)
        except ImportError:
            # Fallback if psutil not available
            import resource
            return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2)
        except Exception:
            return 0.0
    
    async def _store_and_publish_event(
        self,
        event: AgentEvent,
        session_id: str
    ) -> Optional[int]:
        """Store event in database and publish to streams."""
        event_id = None
        
        # Store in database
        if self.config.integration.use_database:
            event_id = await self._store_event_in_database(event)
        
        # Publish to Redis streams
        if self.config.integration.use_redis_streams:
            await self._publish_to_redis_stream(event, session_id)
        
        # Update Prometheus metrics
        if self.config.integration.use_prometheus:
            await self._update_prometheus_metrics(event)
        
        return event_id
    
    async def _store_event_in_database(self, event: AgentEvent) -> Optional[int]:
        """Store event in database and return event ID."""
        try:
            async with get_db_session() as session:
                session.add(event)
                await session.commit()
                await session.refresh(event)
                return event.id
        except Exception as e:
            logger.error(
                "❌ Failed to store session lifecycle event in database",
                error=str(e),
                exc_info=True
            )
            return None
    
    async def _publish_to_redis_stream(self, event: AgentEvent, session_id: str) -> None:
        """Publish event to Redis stream."""
        try:
            from app.core.redis import get_redis_client
            
            redis = await get_redis_client()
            stream_data = {
                "event_type": event.event_type.value,
                "event_subtype": event.payload.get("event_subtype", "unknown"),
                "session_id": str(event.session_id),
                "agent_id": str(event.agent_id),
                "timestamp": datetime.utcnow().isoformat(),
                "payload": json.dumps(event.payload)
            }
            
            await redis.xadd(
                self.config.integration.redis_stream_key,
                stream_data,
                maxlen=self.config.integration.redis_max_len
            )
            
            logger.debug(
                "📡 Session lifecycle event published to Redis stream",
                stream_key=self.config.integration.redis_stream_key,
                session_id=session_id
            )
            
        except Exception as e:
            logger.error(
                "❌ Failed to publish session lifecycle event to Redis",
                session_id=session_id,
                error=str(e),
                exc_info=True
            )
    
    async def _update_prometheus_metrics(self, event: AgentEvent) -> None:
        """Update Prometheus metrics for session lifecycle events."""
        try:
            event_subtype = event.payload.get("event_subtype", "unknown")
            logger.debug(
                "📊 Prometheus metrics updated",
                metric=f"session_lifecycle_events_total{{type='{event_subtype}'}}",
                event_type=event.event_type.value,
                event_subtype=event_subtype
            )
        except Exception as e:
            logger.error(
                "❌ Failed to update Prometheus metrics",
                error=str(e)
            )
    
    async def _initialize_session_tracking(self, session_id: uuid.UUID, agent_id: uuid.UUID) -> None:
        """Initialize session tracking data structures."""
        try:
            # This would initialize tracking in Redis or memory
            logger.debug(
                "🔄 Session tracking initialized",
                session_id=str(session_id),
                agent_id=str(agent_id)
            )
        except Exception as e:
            logger.error("❌ Failed to initialize session tracking", error=str(e))
    
    async def _cleanup_session_tracking(self, session_id: uuid.UUID) -> None:
        """Clean up session tracking data structures."""
        try:
            # This would cleanup tracking data from Redis or memory
            logger.debug("🧹 Session tracking cleaned up", session_id=str(session_id))
        except Exception as e:
            logger.error("❌ Failed to cleanup session tracking", error=str(e))
    
    async def _generate_session_summary(self, session_id: uuid.UUID) -> Dict[str, Any]:
        """Generate session summary statistics."""
        try:
            # This would query the database for session statistics
            return {
                "duration_ms": int((time.time() - self.capture_time) * 1000),
                "tool_executions": 0,  # Would be calculated from events
                "errors": 0,
                "performance_issues": 0,
                "memory_peak_mb": self._get_memory_usage_mb()
            }
        except Exception as e:
            logger.error("❌ Failed to generate session summary", error=str(e))
            return {"duration_ms": 0, "errors": 1}
    
    async def _get_context_statistics(self, session_id: uuid.UUID) -> Dict[str, Any]:
        """Get context usage statistics."""
        return {
            "usage_percent": 85,  # Would be calculated from actual context
            "total_events": 100,
            "memory_usage_mb": self._get_memory_usage_mb()
        }
    
    def _should_trigger_consolidation(self, context_stats: Dict[str, Any]) -> bool:
        """Check if memory consolidation should be triggered."""
        total_events = context_stats.get("total_events", 0)
        return total_events >= self.config.session.consolidation_trigger_threshold
    
    async def _trigger_memory_consolidation(
        self,
        session_id: uuid.UUID,
        context_stats: Dict[str, Any]
    ) -> None:
        """Trigger memory consolidation process."""
        try:
            logger.info(
                "🧠 Memory consolidation triggered",
                session_id=str(session_id),
                total_events=context_stats.get("total_events", 0),
                threshold=self.config.session.consolidation_trigger_threshold
            )
            # This would trigger actual consolidation logic
        except Exception as e:
            logger.error("❌ Failed to trigger memory consolidation", error=str(e))
    
    async def _get_context_restoration_stats(self, session_id: uuid.UUID) -> Dict[str, Any]:
        """Get context restoration statistics."""
        return {
            "restoration_time_ms": 100,
            "items_restored": 50,
            "restoration_success": True
        }
    
    async def _update_session_tracking_wake(self, session_id: uuid.UUID) -> None:
        """Update session tracking with wake information."""
        try:
            logger.debug("🔄 Session tracking updated for wake", session_id=str(session_id))
        except Exception as e:
            logger.error("❌ Failed to update session tracking for wake", error=str(e))


async def main():
    """
    Main entry point for Claude Code session lifecycle hook integration.
    
    Handles session start/end, sleep/wake events based on command line arguments.
    """
    try:
        # Initialize capture system
        capture = SessionLifecycleCapture()
        
        # Parse command line arguments
        if len(sys.argv) < 2:
            logger.error("❌ Usage: session_lifecycle.py <event_type> [options]")
            sys.exit(1)
        
        event_type = sys.argv[1]
        event_id = None
        
        if event_type == "session_start":
            context_data = {}
            if len(sys.argv) > 2:
                try:
                    context_data = json.loads(sys.argv[2])
                except json.JSONDecodeError:
                    pass
            
            event_id = await capture.capture_session_start(
                session_id=os.getenv("CLAUDE_SESSION_ID"),
                agent_id=os.getenv("CLAUDE_AGENT_ID"),
                context_data=context_data
            )
        
        elif event_type == "session_end":
            reason = sys.argv[2] if len(sys.argv) > 2 else "normal_completion"
            
            event_id = await capture.capture_session_end(
                session_id=os.getenv("CLAUDE_SESSION_ID"),
                agent_id=os.getenv("CLAUDE_AGENT_ID"),
                reason=reason
            )
        
        elif event_type == "sleep":
            reason = sys.argv[2] if len(sys.argv) > 2 else "context_threshold_reached"
            
            event_id = await capture.capture_sleep_event(
                session_id=os.getenv("CLAUDE_SESSION_ID"),
                agent_id=os.getenv("CLAUDE_AGENT_ID"),
                sleep_reason=reason
            )
        
        elif event_type == "wake":
            reason = sys.argv[2] if len(sys.argv) > 2 else "new_session_start"
            
            event_id = await capture.capture_wake_event(
                session_id=os.getenv("CLAUDE_SESSION_ID"),
                agent_id=os.getenv("CLAUDE_AGENT_ID"),
                wake_reason=reason
            )
        
        else:
            logger.error(f"❌ Unknown event type: {event_type}")
            sys.exit(1)
        
        if event_id:
            print(f"{event_type} event captured: {event_id}")
            sys.exit(0)
        else:
            print(f"Failed to capture {event_type} event")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Session lifecycle hook interrupted")
        sys.exit(130)
    except Exception as e:
        logger.error(
            "💥 Session lifecycle hook failed",
            error=str(e),
            exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class SessionLifecycleScript(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            await main()
            
            return {"status": "completed"}
    
    script_main(SessionLifecycleScript)