#!/usr/bin/env python3
"""
Post-Tool-Use Hook for LeanVibe Agent Hive 2.0 Observability

Captures and analyzes tool execution completion events for Claude Code integration.
Monitors performance, success/failure rates, and provides comprehensive result logging.
"""

import asyncio
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

# Add parent directories to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.observability import AgentEvent, EventType
from app.core.database import get_db_session
from app.observability.hooks.hooks_config import get_hook_config

logger = structlog.get_logger()


class PostToolUseCapture:
    """
    Post-tool-use event capture and performance analysis system.
    
    Handles the capture of tool execution completion events including
    performance analysis, error tracking, and result processing.
    """
    
    def __init__(self):
        """Initialize post-tool-use capture system."""
        self.config = get_hook_config()
        self.capture_time = time.time()
        
        logger.debug(
            "🔧 PostToolUseCapture initialized",
            config_environment=self.config.environment
        )
    
    async def capture_post_tool_use(
        self,
        tool_name: str,
        success: bool,
        execution_time_ms: Optional[int] = None,
        result: Any = None,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Capture post-tool-use event with comprehensive performance analysis.
        
        Args:
            tool_name: Name of the tool that was executed
            success: Whether the tool execution succeeded
            execution_time_ms: Tool execution time in milliseconds
            result: Tool execution result (if successful)
            error: Error message (if failed)
            error_type: Type/category of error
            session_id: Optional session identifier
            agent_id: Optional agent identifier
            correlation_id: Optional correlation ID for request tracing
            
        Returns:
            Event ID if captured successfully, None otherwise
        """
        if not self.config.should_capture_event("PostToolUse"):
            logger.debug("🚫 PostToolUse event capture disabled")
            return None
        
        try:
            # Generate IDs if not provided
            if session_id is None:
                session_id = self.config.get_session_id()
            if agent_id is None:
                agent_id = self.config.get_agent_id()
            if correlation_id is None:
                correlation_id = str(uuid.uuid4())
            
            # Convert string IDs to UUID objects
            session_uuid = uuid.UUID(session_id) if isinstance(session_id, str) else session_id
            agent_uuid = uuid.UUID(agent_id) if isinstance(agent_id, str) else agent_id
            
            # Sanitize result data for security
            sanitized_result = self._sanitize_result(result)
            
            # Analyze performance
            performance_analysis = self._analyze_performance(
                tool_name=tool_name,
                execution_time_ms=execution_time_ms,
                success=success
            )
            
            # Calculate latency including hook overhead
            hook_latency_ms = int((time.time() - self.capture_time) * 1000)
            total_latency_ms = execution_time_ms + hook_latency_ms if execution_time_ms else hook_latency_ms
            
            # Create event using AgentEvent model
            event = AgentEvent.create_post_tool_use(
                session_id=session_uuid,
                agent_id=agent_uuid,
                tool_name=tool_name,
                success=success,
                result=sanitized_result,
                error=error,
                execution_time_ms=execution_time_ms,
                correlation_id=correlation_id,
                latency_ms=total_latency_ms
            )
            
            # Add additional metadata
            event.payload["hook_version"] = "1.0"
            event.payload["capture_time"] = time.time()
            event.payload["environment"] = self.config.environment
            event.payload["hook_latency_ms"] = hook_latency_ms
            
            # Add error type if provided
            if error_type:
                event.payload["error_type"] = error_type
            
            # Add performance analysis
            event.payload["performance_analysis"] = performance_analysis
            
            # Add result metadata
            if result is not None:
                event.payload["result_metadata"] = self._analyze_result_metadata(result)
            
            # Store event in database if enabled
            event_id = None
            if self.config.integration.use_database:
                event_id = await self._store_event_in_database(event)
            
            # Publish to Redis streams if enabled
            if self.config.integration.use_redis_streams:
                await self._publish_to_redis_stream(event, correlation_id)
            
            # Update Prometheus metrics if enabled
            if self.config.integration.use_prometheus:
                await self._update_prometheus_metrics(
                    tool_name=tool_name,
                    success=success,
                    execution_time_ms=execution_time_ms,
                    performance_analysis=performance_analysis
                )
            
            # Send alerts for performance issues
            if performance_analysis.get("is_slow", False) or not success:
                await self._send_performance_alert(
                    event=event,
                    performance_analysis=performance_analysis
                )
            
            # Send webhooks if configured
            if self.config.integration.webhook_urls:
                asyncio.create_task(self._send_webhooks(event, correlation_id))
            
            # Log based on performance and success
            log_level = self._determine_log_level(success, performance_analysis)
            logger.log(
                log_level,
                "📊 PostToolUse event captured",
                event_id=event_id,
                tool_name=tool_name,
                success=success,
                execution_time_ms=execution_time_ms,
                total_latency_ms=total_latency_ms,
                session_id=str(session_uuid),
                agent_id=str(agent_uuid),
                correlation_id=correlation_id,
                performance_category=performance_analysis.get("category", "normal"),
                error_type=error_type
            )
            
            return str(event_id) if event_id else correlation_id
            
        except Exception as e:
            logger.error(
                "❌ Failed to capture PostToolUse event",
                tool_name=tool_name,
                success=success,
                session_id=str(session_id) if session_id else None,
                agent_id=str(agent_id) if agent_id else None,
                correlation_id=correlation_id,
                error=str(e),
                exc_info=True
            )
            return None
    
    def _sanitize_result(self, result: Any) -> Any:
        """
        Sanitize result data by removing or redacting sensitive information.
        
        Args:
            result: Original result data
            
        Returns:
            Sanitized result data
        """
        if not self.config.security.sanitize_sensitive_data:
            return result
        
        import re
        
        # Handle different result types
        if isinstance(result, str):
            sanitized_result = result
            for pattern in self.config.security.sensitive_patterns:
                sanitized_result = re.sub(pattern, r"[REDACTED]", sanitized_result, flags=re.IGNORECASE)
            
            # Truncate large results
            if len(sanitized_result) > 10000:
                return sanitized_result[:10000] + "... (truncated)"
            
            return sanitized_result
        
        elif isinstance(result, dict):
            sanitized = {}
            for key, value in result.items():
                if any(pattern in key.lower() for pattern in ["password", "token", "key", "secret"]):
                    sanitized[key] = "[REDACTED]"
                else:
                    sanitized[key] = self._sanitize_result(value)
            return sanitized
        
        elif isinstance(result, list):
            return [self._sanitize_result(item) for item in result[:100]]  # Limit list size
        
        else:
            return result
    
    def _analyze_performance(
        self,
        tool_name: str,
        execution_time_ms: Optional[int],
        success: bool
    ) -> Dict[str, Any]:
        """
        Analyze tool performance and categorize execution.
        
        Args:
            tool_name: Name of the executed tool
            execution_time_ms: Execution time in milliseconds
            success: Whether execution was successful
            
        Returns:
            Performance analysis dictionary
        """
        analysis = {
            "category": "normal",
            "is_slow": False,
            "is_very_slow": False,
            "success": success
        }
        
        if execution_time_ms is None:
            analysis["category"] = "no_timing"
            return analysis
        
        # Categorize performance based on thresholds
        if execution_time_ms > self.config.performance.very_slow_tool_threshold_ms:
            analysis["category"] = "very_slow"
            analysis["is_very_slow"] = True
            analysis["is_slow"] = True
        elif execution_time_ms > self.config.performance.slow_tool_threshold_ms:
            analysis["category"] = "slow"
            analysis["is_slow"] = True
        elif execution_time_ms < 100:
            analysis["category"] = "fast"
        
        # Add performance recommendations
        if analysis["is_very_slow"]:
            analysis["recommendation"] = f"Tool '{tool_name}' is very slow ({execution_time_ms}ms). Consider optimization."
        elif analysis["is_slow"]:
            analysis["recommendation"] = f"Tool '{tool_name}' is slow ({execution_time_ms}ms). Monitor for patterns."
        
        # Add success rate context
        analysis["execution_time_ms"] = execution_time_ms
        analysis["timestamp"] = datetime.utcnow().isoformat()
        
        return analysis
    
    def _analyze_result_metadata(self, result: Any) -> Dict[str, Any]:
        """
        Analyze result metadata for monitoring and optimization.
        
        Args:
            result: Tool execution result
            
        Returns:
            Result metadata dictionary
        """
        metadata = {
            "type": type(result).__name__,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if isinstance(result, str):
            metadata["length"] = len(result)
            metadata["is_empty"] = len(result) == 0
            metadata["is_json"] = result.strip().startswith(("{", "["))
        elif isinstance(result, (list, tuple)):
            metadata["length"] = len(result)
            metadata["is_empty"] = len(result) == 0
        elif isinstance(result, dict):
            metadata["keys_count"] = len(result.keys())
            metadata["is_empty"] = len(result) == 0
            metadata["top_level_keys"] = list(result.keys())[:10]  # Limit for payload size
        elif result is None:
            metadata["is_null"] = True
        
        # Estimate memory usage
        metadata["estimated_size_bytes"] = sys.getsizeof(result)
        
        return metadata
    
    def _determine_log_level(
        self,
        success: bool,
        performance_analysis: Dict[str, Any]
    ) -> str:
        """Determine appropriate log level based on execution result and performance."""
        if not success:
            return "error"
        elif performance_analysis.get("is_very_slow", False):
            return "warning"
        elif performance_analysis.get("is_slow", False):
            return "info"
        else:
            return "debug"
    
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
                "❌ Failed to store PostToolUse event in database",
                error=str(e),
                exc_info=True
            )
            return None
    
    async def _publish_to_redis_stream(self, event: AgentEvent, correlation_id: str) -> None:
        """Publish event to Redis stream for real-time processing."""
        try:
            from app.core.redis import get_redis_client
            
            redis = await get_redis_client()
            stream_data = {
                "event_type": event.event_type.value,
                "session_id": str(event.session_id),
                "agent_id": str(event.agent_id),
                "tool_name": event.payload.get("tool_name", "unknown"),
                "success": event.payload.get("success", False),
                "execution_time_ms": event.payload.get("execution_time_ms"),
                "correlation_id": correlation_id,
                "timestamp": datetime.utcnow().isoformat(),
                "performance_category": event.payload.get("performance_analysis", {}).get("category", "unknown"),
                "payload": json.dumps(event.payload)
            }
            
            await redis.xadd(
                self.config.integration.redis_stream_key,
                stream_data,
                maxlen=self.config.integration.redis_max_len
            )
            
            logger.debug(
                "📡 PostToolUse event published to Redis stream",
                stream_key=self.config.integration.redis_stream_key,
                correlation_id=correlation_id
            )
            
        except Exception as e:
            logger.error(
                "❌ Failed to publish PostToolUse event to Redis",
                correlation_id=correlation_id,
                error=str(e),
                exc_info=True
            )
    
    async def _update_prometheus_metrics(
        self,
        tool_name: str,
        success: bool,
        execution_time_ms: Optional[int],
        performance_analysis: Dict[str, Any]
    ) -> None:
        """Update Prometheus metrics for monitoring dashboards."""
        try:
            # This would integrate with existing Prometheus setup
            metrics_to_update = {
                "post_tool_use_total": 1,
                f"tool_executions_total{{tool_name='{tool_name}',success='{success}'}}": 1,
                f"tool_performance_category_total{{category='{performance_analysis.get('category', 'unknown')}'}}": 1
            }
            
            if execution_time_ms:
                metrics_to_update[f"tool_execution_duration_ms{{tool_name='{tool_name}'}}"] = execution_time_ms
            
            logger.debug(
                "📊 Prometheus metrics updated",
                metrics=list(metrics_to_update.keys()),
                tool_name=tool_name,
                success=success
            )
        except Exception as e:
            logger.error(
                "❌ Failed to update Prometheus metrics",
                tool_name=tool_name,
                error=str(e)
            )
    
    async def _send_performance_alert(
        self,
        event: AgentEvent,
        performance_analysis: Dict[str, Any]
    ) -> None:
        """Send performance alerts for slow tools or failures."""
        try:
            alert_data = {
                "alert_type": "performance" if event.payload.get("success") else "error",
                "tool_name": event.payload.get("tool_name"),
                "session_id": str(event.session_id),
                "agent_id": str(event.agent_id),
                "execution_time_ms": event.payload.get("execution_time_ms"),
                "performance_category": performance_analysis.get("category"),
                "recommendation": performance_analysis.get("recommendation"),
                "timestamp": datetime.utcnow().isoformat(),
                "severity": "critical" if performance_analysis.get("is_very_slow") else "warning"
            }
            
            logger.warning(
                "⚠️  Performance alert triggered",
                **alert_data
            )
            
            # Could integrate with alerting systems like PagerDuty, Slack, etc.
            
        except Exception as e:
            logger.error(
                "❌ Failed to send performance alert",
                error=str(e)
            )
    
    async def _send_webhooks(self, event: AgentEvent, correlation_id: str) -> None:
        """Send webhook notifications to configured URLs."""
        import aiohttp
        
        webhook_data = {
            "event_type": "PostToolUse",
            "tool_name": event.payload.get("tool_name"),
            "success": event.payload.get("success"),
            "execution_time_ms": event.payload.get("execution_time_ms"),
            "performance_category": event.payload.get("performance_analysis", {}).get("category"),
            "session_id": str(event.session_id),
            "agent_id": str(event.agent_id),
            "correlation_id": correlation_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        async with aiohttp.ClientSession() as session:
            for webhook_url in self.config.integration.webhook_urls:
                try:
                    async with session.post(
                        webhook_url,
                        json=webhook_data,
                        timeout=aiohttp.ClientTimeout(
                            total=self.config.integration.webhook_timeout_seconds
                        )
                    ) as response:
                        if response.status == 200:
                            logger.debug(
                                "🔔 Webhook sent successfully",
                                webhook_url=webhook_url,
                                correlation_id=correlation_id
                            )
                        else:
                            logger.warning(
                                "⚠️  Webhook failed",
                                webhook_url=webhook_url,
                                status_code=response.status,
                                correlation_id=correlation_id
                            )
                            
                except Exception as e:
                    logger.error(
                        "❌ Failed to send webhook",
                        webhook_url=webhook_url,
                        correlation_id=correlation_id,
                        error=str(e)
                    )


async def main():
    """
    Main entry point for Claude Code hook integration.
    
    Reads tool execution results from command line arguments or stdin
    and processes the post-tool-use event.
    """
    try:
        # Initialize capture system
        capture = PostToolUseCapture()
        
        # Parse command line arguments or read from stdin
        if len(sys.argv) > 1:
            # Command line mode: tool_name success=true/false execution_time_ms=123 result="data"
            tool_name = sys.argv[1]
            success = True
            execution_time_ms = None
            result = None
            error = None
            error_type = None
            
            for arg in sys.argv[2:]:
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    if key == "success":
                        success = value.lower() in ("true", "1", "yes")
                    elif key == "execution_time_ms":
                        execution_time_ms = int(value)
                    elif key == "result":
                        try:
                            result = json.loads(value)
                        except json.JSONDecodeError:
                            result = value
                    elif key == "error":
                        error = value
                    elif key == "error_type":
                        error_type = value
        else:
            # Stdin mode: read JSON data
            input_data = sys.stdin.read().strip()
            if input_data:
                data = json.loads(input_data)
                tool_name = data.get("tool_name", "unknown")
                success = data.get("success", True)
                execution_time_ms = data.get("execution_time_ms")
                result = data.get("result")
                error = data.get("error")
                error_type = data.get("error_type")
            else:
                logger.error("❌ No input data provided")
                sys.exit(1)
        
        # Capture the post-tool-use event
        event_id = await capture.capture_post_tool_use(
            tool_name=tool_name,
            success=success,
            execution_time_ms=execution_time_ms,
            result=result,
            error=error,
            error_type=error_type,
            session_id=os.getenv("CLAUDE_SESSION_ID"),
            agent_id=os.getenv("CLAUDE_AGENT_ID")
        )
        
        if event_id:
            print(f"PostToolUse event captured: {event_id}")
            sys.exit(0)
        else:
            print("Failed to capture PostToolUse event")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 PostToolUse hook interrupted")
        sys.exit(130)
    except Exception as e:
        logger.error(
            "💥 PostToolUse hook failed",
            error=str(e),
            exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class PostToolUseScript(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            import os
            await main()
            
            return {"status": "completed"}
    
    script_main(PostToolUseScript)