"""
Enhanced Hook Event Processor for LeanVibe Agent Hive 2.0 Observability System

Provides automatic event capture from Claude Code hooks with real-time streaming,
PII redaction, performance monitoring, and comprehensive security filtering.
"""

import asyncio
import json
import re
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Pattern, Protocol, Union

import structlog
from redis.asyncio import Redis

from app.core.event_processor import EventStreamProcessor, get_event_processor
from app.models.observability import EventType

logger = structlog.get_logger()


class PIIRedactor:
    """Advanced PII redaction with pattern recognition and context awareness."""
    
    def __init__(self):
        """Initialize PII redaction patterns."""
        # Compiled regex patterns for efficient matching
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
            'ssn': re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            'credit_card': re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'),
            'api_key': re.compile(r'\b[A-Za-z0-9]{32,}\b'),
            'jwt_token': re.compile(r'\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b'),
            'ip_address': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            'file_path': re.compile(r'(?:/Users/[^/\s]+|/home/[^/\s]+|C:\\Users\\[^\\s]+)'),
            'aws_key': re.compile(r'\bAKIA[0-9A-Z]{16}\b'),
            'github_token': re.compile(r'\bghp_[A-Za-z0-9]{36}\b'),
            'password_field': re.compile(r'(?i)(?:password|passwd|pwd|secret|token|key|auth)["\']?\s*[:=]\s*["\']?([^"\'\s,}]+)')
        }
        
        # Sensitive field names
        self.sensitive_fields = {
            'password', 'passwd', 'pwd', 'secret', 'token', 'key', 'auth', 
            'credential', 'authorization', 'x-api-key', 'api_key', 'access_token',
            'refresh_token', 'session_token', 'csrf_token', 'private_key',
            'client_secret', 'database_url', 'connection_string'
        }
        
        # PII field names
        self.pii_fields = {
            'email', 'phone', 'ssn', 'credit_card', 'social_security',
            'driver_license', 'passport', 'first_name', 'last_name',
            'full_name', 'address', 'zipcode', 'postal_code', 'date_of_birth'
        }
    
    def redact_data(self, data: Any, context: Optional[str] = None) -> Any:
        """
        Recursively redact PII and sensitive data from any data structure.
        
        Args:
            data: Data to redact (dict, list, str, or other)
            context: Context string for better redaction decisions
            
        Returns:
            Data with sensitive information redacted
        """
        if isinstance(data, dict):
            return self._redact_dict(data, context)
        elif isinstance(data, list):
            return self._redact_list(data, context)
        elif isinstance(data, str):
            return self._redact_string(data, context)
        else:
            return data
    
    def _redact_dict(self, data: Dict[str, Any], context: Optional[str] = None) -> Dict[str, Any]:
        """Redact sensitive data from dictionary."""
        redacted = {}
        
        for key, value in data.items():
            key_lower = key.lower()
            
            # Check if field name indicates sensitive data
            if any(sensitive in key_lower for sensitive in self.sensitive_fields):
                redacted[key] = "[REDACTED]"
            elif any(pii in key_lower for pii in self.pii_fields):
                redacted[key] = "[PII_REDACTED]"
            else:
                # Recursively redact value
                redacted[key] = self.redact_data(value, f"{context}.{key}" if context else key)
        
        return redacted
    
    def _redact_list(self, data: List[Any], context: Optional[str] = None) -> List[Any]:
        """Redact sensitive data from list."""
        # Limit list size to prevent performance issues
        max_items = 100
        if len(data) > max_items:
            redacted_items = [self.redact_data(item, f"{context}[{i}]" if context else f"item_{i}") 
                            for i, item in enumerate(data[:max_items])]
            redacted_items.append(f"[TRUNCATED: {len(data) - max_items} more items]")
            return redacted_items
        else:
            return [self.redact_data(item, f"{context}[{i}]" if context else f"item_{i}") 
                   for i, item in enumerate(data)]
    
    def _redact_string(self, data: str, context: Optional[str] = None) -> str:
        """Redact sensitive patterns from string."""
        if not data or not isinstance(data, str):
            return data
        
        # Truncate very large strings
        if len(data) > 50000:
            data = data[:50000] + "... [TRUNCATED]"
        
        redacted = data
        
        # Apply pattern-based redaction
        for pattern_name, pattern in self.patterns.items():
            if pattern_name == 'password_field':
                # Special handling for password fields
                redacted = pattern.sub(lambda m: f"{m.group(0).split('=')[0]}=[REDACTED]", redacted)
            else:
                replacement = f"[{pattern_name.upper()}_REDACTED]"
                redacted = pattern.sub(replacement, redacted)
        
        return redacted


class PerformanceMonitor:
    """Performance monitoring for hook event processing."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = {
            'events_processed': 0,
            'events_failed': 0,
            'total_processing_time_ms': 0,
            'max_processing_time_ms': 0,
            'avg_processing_time_ms': 0,
            'redaction_time_ms': 0,
            'api_time_ms': 0
        }
        self.start_time = time.time()
        self.processing_times = []
    
    def record_event_processing(self, processing_time_ms: float, redaction_time_ms: float = 0, api_time_ms: float = 0):
        """Record metrics for event processing."""
        self.metrics['events_processed'] += 1
        self.metrics['total_processing_time_ms'] += processing_time_ms
        self.metrics['redaction_time_ms'] += redaction_time_ms
        self.metrics['api_time_ms'] += api_time_ms
        
        # Update max processing time
        if processing_time_ms > self.metrics['max_processing_time_ms']:
            self.metrics['max_processing_time_ms'] = processing_time_ms
        
        # Keep rolling window of processing times
        self.processing_times.append(processing_time_ms)
        if len(self.processing_times) > 1000:
            self.processing_times = self.processing_times[-1000:]
        
        # Update average
        if self.metrics['events_processed'] > 0:
            self.metrics['avg_processing_time_ms'] = (
                self.metrics['total_processing_time_ms'] / self.metrics['events_processed']
            )
    
    def record_event_failure(self):
        """Record event processing failure."""
        self.metrics['events_failed'] += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        uptime_seconds = time.time() - self.start_time
        
        # Calculate percentiles
        percentiles = {}
        if self.processing_times:
            sorted_times = sorted(self.processing_times)
            percentiles = {
                'p50': sorted_times[len(sorted_times) // 2],
                'p95': sorted_times[int(len(sorted_times) * 0.95)],
                'p99': sorted_times[int(len(sorted_times) * 0.99)]
            }
        
        return {
            'uptime_seconds': uptime_seconds,
            'events_per_second': self.metrics['events_processed'] / max(uptime_seconds, 1),
            'success_rate': (
                (self.metrics['events_processed'] / 
                 max(self.metrics['events_processed'] + self.metrics['events_failed'], 1)) * 100
            ),
            'processing_time_percentiles': percentiles,
            **self.metrics
        }
    
    def is_performance_degraded(self) -> Dict[str, Any]:
        """Check if performance is degraded."""
        issues = []
        warnings = []
        
        # Check average processing time
        if self.metrics['avg_processing_time_ms'] > 150:  # PRD requirement: <150ms
            issues.append(f"Average processing time ({self.metrics['avg_processing_time_ms']:.1f}ms) exceeds 150ms threshold")
        elif self.metrics['avg_processing_time_ms'] > 100:
            warnings.append(f"Average processing time ({self.metrics['avg_processing_time_ms']:.1f}ms) approaching threshold")
        
        # Check failure rate
        total_events = self.metrics['events_processed'] + self.metrics['events_failed']
        if total_events > 0:
            failure_rate = (self.metrics['events_failed'] / total_events) * 100
            if failure_rate > 5:
                issues.append(f"High failure rate: {failure_rate:.1f}%")
            elif failure_rate > 2:
                warnings.append(f"Elevated failure rate: {failure_rate:.1f}%")
        
        return {
            'is_degraded': len(issues) > 0,
            'issues': issues,
            'warnings': warnings
        }


class HookEventProcessor:
    """Enhanced hook event processor with automatic capture, PII redaction, and performance monitoring."""
    
    def __init__(
        self,
        redis_client: Redis,
        event_processor: Optional[EventStreamProcessor] = None,
        enable_pii_redaction: bool = True,
        enable_performance_monitoring: bool = True
    ):
        """
        Initialize HookEventProcessor.
        
        Args:
            redis_client: Redis client for real-time streaming
            event_processor: Event processor for database persistence
            enable_pii_redaction: Enable PII redaction
            enable_performance_monitoring: Enable performance monitoring
        """
        self.redis_client = redis_client
        self.event_processor = event_processor or get_event_processor()
        self.pii_redactor = PIIRedactor() if enable_pii_redaction else None
        self.performance_monitor = PerformanceMonitor() if enable_performance_monitoring else None
        
        # Real-time streaming configuration
        self.observability_stream = "observability_events"
        self.websocket_stream = "websocket_events"
        
        logger.info(
            "🔧 HookEventProcessor initialized",
            pii_redaction=enable_pii_redaction,
            performance_monitoring=enable_performance_monitoring
        )
    
    async def process_pre_tool_use(self, event_data: Dict[str, Any]) -> Optional[str]:
        """
        Process PreToolUse hook event with security filtering and performance monitoring.
        
        Args:
            event_data: Raw event data from hook
            
        Returns:
            Event ID if processed successfully
        """
        start_time = time.time()
        redaction_start = time.time()
        
        try:
            # Extract and validate required fields
            session_id = uuid.UUID(event_data.get("session_id"))
            agent_id = uuid.UUID(event_data.get("agent_id"))
            tool_name = event_data.get("tool_name", "unknown")
            parameters = event_data.get("parameters", {})
            
            # PII redaction
            safe_parameters = parameters
            if self.pii_redactor:
                safe_parameters = self.pii_redactor.redact_data(parameters, "tool_parameters")
            
            redaction_time = (time.time() - redaction_start) * 1000
            
            # Prepare enhanced event payload
            payload = {
                "tool_name": tool_name,
                "parameters": safe_parameters,
                "correlation_id": event_data.get("correlation_id"),
                "timestamp": event_data.get("timestamp", datetime.utcnow().isoformat()),
                "context": {
                    "parameter_count": len(parameters),
                    "parameter_size_bytes": len(json.dumps(parameters, default=str)),
                    "redacted": self.pii_redactor is not None,
                    "redaction_applied": safe_parameters != parameters if self.pii_redactor else False,
                    "hook_version": "2.0"
                }
            }
            
            # Process through event processor
            api_start = time.time()
            event_id = None
            if self.event_processor:
                event_id = await self.event_processor.process_event(
                    session_id=session_id,
                    agent_id=agent_id,
                    event_type=EventType.PRE_TOOL_USE,
                    payload=payload
                )
            
            api_time = (time.time() - api_start) * 1000
            
            # Stream to real-time channels
            await self._stream_event_real_time({
                "event_type": "PRE_TOOL_USE",
                "session_id": str(session_id),
                "agent_id": str(agent_id),
                "tool_name": tool_name,
                "timestamp": payload["timestamp"],
                "event_id": event_id
            })
            
            # Record performance metrics
            total_time = (time.time() - start_time) * 1000
            if self.performance_monitor:
                self.performance_monitor.record_event_processing(total_time, redaction_time, api_time)
            
            logger.debug(
                "✅ PreToolUse event processed",
                event_id=event_id,
                tool_name=tool_name,
                processing_time_ms=total_time,
                redaction_time_ms=redaction_time
            )
            
            return event_id
            
        except Exception as e:
            if self.performance_monitor:
                self.performance_monitor.record_event_failure()
            
            logger.error(
                "❌ Failed to process PreToolUse event",
                error=str(e),
                event_data=event_data,
                exc_info=True
            )
            return None
    
    async def process_post_tool_use(self, event_data: Dict[str, Any]) -> Optional[str]:
        """
        Process PostToolUse hook event with result analysis and performance metrics.
        
        Args:
            event_data: Raw event data from hook
            
        Returns:
            Event ID if processed successfully
        """
        start_time = time.time()
        redaction_start = time.time()
        
        try:
            # Extract and validate required fields
            session_id = uuid.UUID(event_data.get("session_id"))
            agent_id = uuid.UUID(event_data.get("agent_id"))
            tool_name = event_data.get("tool_name", "unknown")
            success = event_data.get("success", True)
            result = event_data.get("result")
            error = event_data.get("error")
            execution_time_ms = event_data.get("execution_time_ms")
            
            # PII redaction for results
            safe_result = result
            safe_error = error
            if self.pii_redactor:
                safe_result = self.pii_redactor.redact_data(result, "tool_result")
                safe_error = self.pii_redactor.redact_data(error, "tool_error") if error else None
            
            redaction_time = (time.time() - redaction_start) * 1000
            
            # Enhanced performance analysis
            performance_analysis = self._analyze_tool_performance(
                tool_name, execution_time_ms, result, success
            )
            
            # Prepare enhanced event payload
            payload = {
                "tool_name": tool_name,
                "success": success,
                "result": safe_result,
                "error": safe_error,
                "correlation_id": event_data.get("correlation_id"),
                "execution_time_ms": execution_time_ms,
                "timestamp": event_data.get("timestamp", datetime.utcnow().isoformat()),
                "performance": performance_analysis,
                "context": {
                    "result_truncated": self._is_truncated(result, safe_result),
                    "result_redacted": safe_result != result if self.pii_redactor else False,
                    "error_redacted": safe_error != error if self.pii_redactor and error else False,
                    "has_error": error is not None,
                    "error_type": type(error).__name__ if error else None,
                    "result_size_bytes": len(json.dumps(safe_result, default=str)) if safe_result else 0,
                    "hook_version": "2.0"
                }
            }
            
            # Process through event processor
            api_start = time.time()
            event_id = None
            if self.event_processor:
                event_id = await self.event_processor.process_event(
                    session_id=session_id,
                    agent_id=agent_id,
                    event_type=EventType.POST_TOOL_USE,
                    payload=payload,
                    latency_ms=execution_time_ms
                )
            
            api_time = (time.time() - api_start) * 1000
            
            # Stream to real-time channels
            await self._stream_event_real_time({
                "event_type": "POST_TOOL_USE",
                "session_id": str(session_id),
                "agent_id": str(agent_id),
                "tool_name": tool_name,
                "success": success,
                "execution_time_ms": execution_time_ms,
                "performance_score": performance_analysis.get("performance_score"),
                "timestamp": payload["timestamp"],
                "event_id": event_id
            })
            
            # Record performance metrics
            total_time = (time.time() - start_time) * 1000
            if self.performance_monitor:
                self.performance_monitor.record_event_processing(total_time, redaction_time, api_time)
            
            logger.debug(
                "✅ PostToolUse event processed",
                event_id=event_id,
                tool_name=tool_name,
                success=success,
                processing_time_ms=total_time,
                performance_score=performance_analysis.get("performance_score")
            )
            
            return event_id
            
        except Exception as e:
            if self.performance_monitor:
                self.performance_monitor.record_event_failure()
            
            logger.error(
                "❌ Failed to process PostToolUse event",
                error=str(e),
                event_data=event_data,
                exc_info=True
            )
            return None
    
    async def process_error_event(self, event_data: Dict[str, Any]) -> Optional[str]:
        """
        Process error hook event with classification and context preservation.
        
        Args:
            event_data: Raw event data from hook
            
        Returns:
            Event ID if processed successfully
        """
        start_time = time.time()
        
        try:
            # Extract and validate required fields
            session_id = uuid.UUID(event_data.get("session_id"))
            agent_id = uuid.UUID(event_data.get("agent_id"))
            error_type = event_data.get("error_type", "UnknownError")
            error_message = event_data.get("error_message", "")
            
            # PII redaction for error data
            safe_error_message = error_message
            safe_stack_trace = event_data.get("stack_trace")
            safe_context = event_data.get("context", {})
            
            if self.pii_redactor:
                safe_error_message = self.pii_redactor.redact_data(error_message, "error_message")
                safe_stack_trace = self.pii_redactor.redact_data(safe_stack_trace, "stack_trace") if safe_stack_trace else None
                safe_context = self.pii_redactor.redact_data(safe_context, "error_context")
            
            # Prepare enhanced event payload
            payload = {
                "error_type": error_type,
                "error_message": safe_error_message,
                "stack_trace": safe_stack_trace,
                "context": safe_context,
                "classification": event_data.get("classification", {}),
                "correlation_id": event_data.get("correlation_id"),
                "timestamp": event_data.get("timestamp", datetime.utcnow().isoformat()),
                "system_info": event_data.get("system_info", {}),
                "metadata": {
                    "sanitized": True,
                    "redacted": self.pii_redactor is not None,
                    "hook_version": "2.0",
                    **event_data.get("metadata", {})
                }
            }
            
            # Process through event processor
            event_id = None
            if self.event_processor:
                event_id = await self.event_processor.process_event(
                    session_id=session_id,
                    agent_id=agent_id,
                    event_type=EventType.ERROR,
                    payload=payload
                )
            
            # Stream to real-time channels with high priority
            await self._stream_event_real_time({
                "event_type": "ERROR",
                "session_id": str(session_id),
                "agent_id": str(agent_id),
                "error_type": error_type,
                "severity": event_data.get("classification", {}).get("severity", "medium"),
                "is_critical": event_data.get("classification", {}).get("is_critical", False),
                "timestamp": payload["timestamp"],
                "event_id": event_id
            }, priority="high")
            
            # Record performance metrics
            total_time = (time.time() - start_time) * 1000
            if self.performance_monitor:
                self.performance_monitor.record_event_processing(total_time)
            
            logger.info(
                "🚨 Error event processed",
                event_id=event_id,
                error_type=error_type,
                severity=event_data.get("classification", {}).get("severity"),
                processing_time_ms=total_time
            )
            
            return event_id
            
        except Exception as e:
            if self.performance_monitor:
                self.performance_monitor.record_event_failure()
            
            logger.error(
                "❌ Failed to process error event",
                error=str(e),
                event_data=event_data,
                exc_info=True
            )
            return None
    
    async def process_agent_lifecycle_event(self, event_data: Dict[str, Any], event_type: EventType) -> Optional[str]:
        """
        Process agent lifecycle events (start/stop) with comprehensive context.
        
        Args:
            event_data: Raw event data from hook
            event_type: AGENT_START or AGENT_STOP
            
        Returns:
            Event ID if processed successfully
        """
        start_time = time.time()
        
        try:
            # Extract and validate required fields
            session_id = uuid.UUID(event_data.get("session_id"))
            agent_id = uuid.UUID(event_data.get("agent_id"))
            agent_name = event_data.get("agent_name", "unknown")
            
            # PII redaction for system information
            safe_payload = event_data.copy()
            if self.pii_redactor:
                # Redact system info and environment data
                if "system_info" in safe_payload:
                    safe_payload["system_info"] = self.pii_redactor.redact_data(
                        safe_payload["system_info"], "system_info"
                    )
                if "environment_info" in safe_payload:
                    safe_payload["environment_info"] = self.pii_redactor.redact_data(
                        safe_payload["environment_info"], "environment_info"
                    )
            
            # Add metadata
            safe_payload["metadata"] = {
                **safe_payload.get("metadata", {}),
                "redacted": self.pii_redactor is not None,
                "hook_version": "2.0",
                "processing_timestamp": datetime.utcnow().isoformat()
            }
            
            # Process through event processor
            event_id = None
            if self.event_processor:
                event_id = await self.event_processor.process_event(
                    session_id=session_id,
                    agent_id=agent_id,
                    event_type=event_type,
                    payload=safe_payload
                )
            
            # Stream to real-time channels
            await self._stream_event_real_time({
                "event_type": event_type.value,
                "session_id": str(session_id),
                "agent_id": str(agent_id),
                "agent_name": agent_name,
                "timestamp": safe_payload.get("timestamp"),
                "event_id": event_id
            })
            
            # Record performance metrics
            total_time = (time.time() - start_time) * 1000
            if self.performance_monitor:
                self.performance_monitor.record_event_processing(total_time)
            
            logger.info(
                f"🔄 {event_type.value} event processed",
                event_id=event_id,
                agent_name=agent_name,
                processing_time_ms=total_time
            )
            
            return event_id
            
        except Exception as e:
            if self.performance_monitor:
                self.performance_monitor.record_event_failure()
            
            logger.error(
                f"❌ Failed to process {event_type.value} event",
                error=str(e),
                event_data=event_data,
                exc_info=True
            )
            return None
    
    def _analyze_tool_performance(
        self, 
        tool_name: str, 
        execution_time_ms: Optional[int], 
        result: Any, 
        success: bool
    ) -> Dict[str, Any]:
        """Analyze tool performance and generate insights."""
        analysis = {
            "performance_score": "good",
            "warnings": [],
            "metrics": {
                "execution_time_ms": execution_time_ms or 0,
                "result_size_bytes": len(json.dumps(result, default=str)) if result else 0
            },
            "insights": []
        }
        
        if not success:
            analysis["performance_score"] = "failed"
            analysis["warnings"].append("Tool execution failed")
            return analysis
        
        # Performance thresholds by tool type
        thresholds = {
            "bash": 10000,  # Shell commands can take longer
            "read": 5000,   # File reads should be fast
            "write": 5000,  # File writes should be fast
            "edit": 3000,   # Edits should be quick
            "default": 2000 # Default threshold
        }
        
        threshold = thresholds.get(tool_name.lower(), thresholds["default"])
        
        if execution_time_ms and execution_time_ms > threshold:
            analysis["performance_score"] = "slow"
            analysis["warnings"].append(f"Execution time ({execution_time_ms}ms) exceeded {threshold}ms threshold")
        
        # Result size analysis
        result_size = analysis["metrics"]["result_size_bytes"]
        if result_size > 100 * 1024:  # 100KB
            analysis["warnings"].append(f"Large result size: {result_size // 1024}KB")
        
        # Tool-specific insights
        if tool_name.lower() in ["read", "write", "edit"] and execution_time_ms:
            if execution_time_ms < 100:
                analysis["insights"].append("Fast file operation")
            elif execution_time_ms > 5000:
                analysis["insights"].append("Consider file size optimization")
        
        return analysis
    
    def _is_truncated(self, original: Any, processed: Any) -> bool:
        """Check if data was truncated during processing."""
        if isinstance(original, str) and isinstance(processed, str):
            return "[TRUNCATED]" in processed
        return False
    
    async def _stream_event_real_time(self, event_data: Dict[str, Any], priority: str = "normal") -> None:
        """Stream event to real-time channels for dashboard updates."""
        try:
            # Add event to observability stream
            stream_data = {
                **event_data,
                "priority": priority,
                "stream_timestamp": datetime.utcnow().isoformat()
            }
            
            await self.redis_client.xadd(
                self.observability_stream,
                stream_data,
                maxlen=10000  # Keep last 10k events
            )
            
            # Also add to WebSocket stream for immediate dashboard updates
            await self.redis_client.xadd(
                self.websocket_stream,
                {"event_data": json.dumps(stream_data, default=str)},
                maxlen=1000  # Smaller buffer for WebSocket
            )
            
        except Exception as e:
            logger.error(
                "❌ Failed to stream event to real-time channels",
                error=str(e),
                event_data=event_data
            )
    
    async def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics for monitoring."""
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "performance": {},
            "health": "healthy"
        }
        
        if self.performance_monitor:
            performance_summary = self.performance_monitor.get_performance_summary()
            degradation_check = self.performance_monitor.is_performance_degraded()
            
            metrics["performance"] = performance_summary
            metrics["degradation"] = degradation_check
            
            if degradation_check["is_degraded"]:
                metrics["health"] = "degraded"
        
        # Add Redis stream information
        try:
            stream_info = await self.redis_client.xinfo_stream(self.observability_stream)
            metrics["stream_info"] = {
                "length": stream_info.get("length", 0),
                "last_generated_id": stream_info.get("last-generated-id", "0-0")
            }
        except Exception as e:
            logger.warning("Failed to get stream info", error=str(e))
            metrics["stream_info"] = {"error": str(e)}
        
        return metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of the hook processing system."""
        health = {
            "status": "healthy",
            "components": {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Check Redis connectivity
        try:
            await self.redis_client.ping()
            health["components"]["redis"] = {"status": "healthy"}
        except Exception as e:
            health["components"]["redis"] = {"status": "unhealthy", "error": str(e)}
            health["status"] = "unhealthy"
        
        # Check event processor
        if self.event_processor:
            try:
                processor_health = await self.event_processor.health_check()
                health["components"]["event_processor"] = processor_health
                if processor_health["status"] != "healthy":
                    health["status"] = "degraded"
            except Exception as e:
                health["components"]["event_processor"] = {"status": "unhealthy", "error": str(e)}
                health["status"] = "unhealthy"
        else:
            health["components"]["event_processor"] = {"status": "not_configured"}
        
        # Check performance
        if self.performance_monitor:
            degradation = self.performance_monitor.is_performance_degraded()
            health["components"]["performance"] = {
                "status": "degraded" if degradation["is_degraded"] else "healthy",
                "metrics": self.performance_monitor.get_performance_summary(),
                "issues": degradation.get("issues", []),
                "warnings": degradation.get("warnings", [])
            }
            
            if degradation["is_degraded"] and health["status"] == "healthy":
                health["status"] = "degraded"
        
        return health


# Global hook event processor instance
_hook_event_processor: Optional[HookEventProcessor] = None


def get_hook_event_processor() -> Optional[HookEventProcessor]:
    """Get the global hook event processor instance."""
    return _hook_event_processor


def set_hook_event_processor(processor: HookEventProcessor) -> None:
    """Set the global hook event processor instance."""
    global _hook_event_processor
    _hook_event_processor = processor
    logger.info("🔗 Global hook event processor set")


async def initialize_hook_event_processor(
    redis_client: Redis,
    event_processor: Optional[EventStreamProcessor] = None
) -> HookEventProcessor:
    """
    Initialize and set the global hook event processor.
    
    Args:
        redis_client: Redis client instance
        event_processor: Event processor instance
        
    Returns:
        HookEventProcessor instance
    """
    processor = HookEventProcessor(
        redis_client=redis_client,
        event_processor=event_processor
    )
    
    set_hook_event_processor(processor)
    
    logger.info("✅ Hook event processor initialized")
    return processor