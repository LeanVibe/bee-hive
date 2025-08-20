#!/usr/bin/env python3
"""
Pre-Tool-Use Hook for LeanVibe Agent Hive 2.0 Observability

Captures and logs tool execution initiation events for Claude Code integration.
Monitors tool parameters, validates inputs, and provides performance baselines.
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


class PreToolUseCapture:
    """
    Pre-tool-use event capture and processing system.
    
    Handles the capture of tool execution initiation events including
    parameter validation, performance monitoring setup, and correlation tracking.
    """
    
    def __init__(self):
        """Initialize pre-tool-use capture system."""
        self.config = get_hook_config()
        self.start_time = time.time()
        
        logger.debug(
            "🔧 PreToolUseCapture initialized",
            config_environment=self.config.environment
        )
    
    async def capture_pre_tool_use(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Capture pre-tool-use event with comprehensive monitoring.
        
        Args:
            tool_name: Name of the tool being executed
            parameters: Tool execution parameters
            session_id: Optional session identifier
            agent_id: Optional agent identifier  
            correlation_id: Optional correlation ID for request tracing
            
        Returns:
            Event ID if captured successfully, None otherwise
        """
        if not self.config.should_capture_event("PreToolUse"):
            logger.debug("🚫 PreToolUse event capture disabled")
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
            
            # Sanitize parameters for security
            sanitized_parameters = self._sanitize_parameters(parameters)
            
            # Validate parameter size
            if not self._validate_payload_size(sanitized_parameters):
                logger.warning(
                    "⚠️  Parameter payload too large, truncating",
                    tool_name=tool_name,
                    original_size=sys.getsizeof(json.dumps(parameters)),
                    max_size=self.config.security.max_payload_size
                )
                sanitized_parameters = self._truncate_parameters(sanitized_parameters)
            
            # Create event using AgentEvent model
            event = AgentEvent.create_pre_tool_use(
                session_id=session_uuid,
                agent_id=agent_uuid,
                tool_name=tool_name,
                parameters=sanitized_parameters,
                correlation_id=correlation_id
            )
            
            # Add additional metadata
            event.payload["hook_version"] = "1.0"
            event.payload["capture_time"] = time.time()
            event.payload["environment"] = self.config.environment
            
            # Add performance monitoring metadata
            event.payload["performance_monitoring"] = {
                "slow_threshold_ms": self.config.performance.slow_tool_threshold_ms,
                "very_slow_threshold_ms": self.config.performance.very_slow_tool_threshold_ms,
                "start_timestamp": time.time()
            }
            
            # Store event in database if enabled
            event_id = None
            if self.config.integration.use_database:
                event_id = await self._store_event_in_database(event)
            
            # Publish to Redis streams if enabled
            if self.config.integration.use_redis_streams:
                await self._publish_to_redis_stream(event, correlation_id)
            
            # Update Prometheus metrics if enabled
            if self.config.integration.use_prometheus:
                await self._update_prometheus_metrics(tool_name, "pre_tool_use")
            
            # Send webhooks if configured
            if self.config.integration.webhook_urls:
                asyncio.create_task(self._send_webhooks(event, correlation_id))
            
            logger.info(
                "📝 PreToolUse event captured",
                event_id=event_id,
                tool_name=tool_name,
                session_id=str(session_uuid),
                agent_id=str(agent_uuid),
                correlation_id=correlation_id,
                parameter_count=len(sanitized_parameters)
            )
            
            return str(event_id) if event_id else correlation_id
            
        except Exception as e:
            logger.error(
                "❌ Failed to capture PreToolUse event",
                tool_name=tool_name,
                session_id=str(session_id) if session_id else None,
                agent_id=str(agent_id) if agent_id else None,
                error=str(e),
                exc_info=True
            )
            return None
    
    def _sanitize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize parameters by removing or redacting sensitive data.
        
        Args:
            parameters: Original parameters dictionary
            
        Returns:
            Sanitized parameters dictionary
        """
        if not self.config.security.sanitize_sensitive_data:
            return parameters
        
        import re
        
        # Deep copy parameters for sanitization
        sanitized = {}
        
        for key, value in parameters.items():
            # Check if key contains sensitive information
            if any(pattern in key.lower() for pattern in ["password", "token", "key", "secret"]):
                sanitized[key] = "[REDACTED]"
                continue
            
            # Sanitize string values using regex patterns
            if isinstance(value, str):
                sanitized_value = value
                for pattern in self.config.security.sensitive_patterns:
                    sanitized_value = re.sub(pattern, r"[REDACTED]", sanitized_value, flags=re.IGNORECASE)
                sanitized[key] = sanitized_value
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_parameters(value)
            elif isinstance(value, list):
                sanitized[key] = [self._sanitize_parameters(item) if isinstance(item, dict) else item for item in value]
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _validate_payload_size(self, parameters: Dict[str, Any]) -> bool:
        """Validate that parameters don't exceed maximum payload size."""
        try:
            payload_size = sys.getsizeof(json.dumps(parameters))
            return payload_size <= self.config.security.max_payload_size
        except Exception:
            return False
    
    def _truncate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Truncate parameters to fit within payload size limits."""
        truncated = {"_truncated": True, "_original_keys": list(parameters.keys())}
        
        # Keep essential keys and truncate others
        essential_keys = ["tool_name", "command", "action", "method"]
        current_size = 0
        
        for key in essential_keys:
            if key in parameters:
                value_str = json.dumps(parameters[key])
                if current_size + len(value_str) < self.config.security.max_payload_size * 0.8:
                    truncated[key] = parameters[key]
                    current_size += len(value_str)
        
        return truncated
    
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
                "❌ Failed to store PreToolUse event in database",
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
                "correlation_id": correlation_id,
                "timestamp": datetime.utcnow().isoformat(),
                "payload": json.dumps(event.payload)
            }
            
            await redis.xadd(
                self.config.integration.redis_stream_key,
                stream_data,
                maxlen=self.config.integration.redis_max_len
            )
            
            logger.debug(
                "📡 PreToolUse event published to Redis stream",
                stream_key=self.config.integration.redis_stream_key,
                correlation_id=correlation_id
            )
            
        except Exception as e:
            logger.error(
                "❌ Failed to publish PreToolUse event to Redis",
                correlation_id=correlation_id,
                error=str(e),
                exc_info=True
            )
    
    async def _update_prometheus_metrics(self, tool_name: str, event_type: str) -> None:
        """Update Prometheus metrics for monitoring."""
        try:
            # This would integrate with existing Prometheus setup
            # For now, just log the metric update
            logger.debug(
                "📊 Prometheus metrics updated",
                metric="pre_tool_use_total",
                tool_name=tool_name,
                event_type=event_type
            )
        except Exception as e:
            logger.error(
                "❌ Failed to update Prometheus metrics",
                tool_name=tool_name,
                error=str(e)
            )
    
    async def _send_webhooks(self, event: AgentEvent, correlation_id: str) -> None:
        """Send webhook notifications to configured URLs."""
        import aiohttp
        
        webhook_data = {
            "event_type": "PreToolUse",
            "tool_name": event.payload.get("tool_name"),
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
    
    Reads tool execution data from command line arguments or stdin
    and processes the pre-tool-use event.
    """
    try:
        # Initialize capture system
        capture = PreToolUseCapture()
        
        # Parse command line arguments or read from stdin
        if len(sys.argv) > 1:
            # Command line mode: tool_name param1=value1 param2=value2
            tool_name = sys.argv[1]
            parameters = {}
            
            for arg in sys.argv[2:]:
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    # Try to parse as JSON, fallback to string
                    try:
                        parameters[key] = json.loads(value)
                    except json.JSONDecodeError:
                        parameters[key] = value
                else:
                    parameters[arg] = True
        else:
            # Stdin mode: read JSON data
            input_data = sys.stdin.read().strip()
            if input_data:
                data = json.loads(input_data)
                tool_name = data.get("tool_name", "unknown")
                parameters = data.get("parameters", {})
            else:
                logger.error("❌ No input data provided")
                sys.exit(1)
        
        # Capture the pre-tool-use event
        event_id = await capture.capture_pre_tool_use(
            tool_name=tool_name,
            parameters=parameters,
            session_id=os.getenv("CLAUDE_SESSION_ID"),
            agent_id=os.getenv("CLAUDE_AGENT_ID")
        )
        
        if event_id:
            print(f"PreToolUse event captured: {event_id}")
            sys.exit(0)
        else:
            print("Failed to capture PreToolUse event")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 PreToolUse hook interrupted")
        sys.exit(130)
    except Exception as e:
        logger.error(
            "💥 PreToolUse hook failed",
            error=str(e),
            exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class PreToolUseScript(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            import os
            await main()
            
            return {"status": "completed"}
    
    script_main(PreToolUseScript)