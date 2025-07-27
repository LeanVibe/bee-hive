"""
External Hook System for LeanVibe Agent Hive 2.0

Provides secure integration with external monitoring systems, webhook endpoints,
and third-party observability platforms. Includes security validation,
audit trails, and rate limiting for production use.
"""

import asyncio
import hashlib
import hmac
import json
import time
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

import aiohttp
import structlog
from pydantic import BaseModel, Field, HttpUrl, validator

from app.observability.hooks import HookInterceptor

logger = structlog.get_logger()

class HookType(str, Enum):
    """Types of external hooks."""
    WEBHOOK = "webhook"
    PROMETHEUS = "prometheus"
    GRAFANA = "grafana"
    SLACK = "slack"
    DISCORD = "discord"
    PAGERDUTY = "pagerduty"
    DATADOG = "datadog"
    NEWRELIC = "newrelic"
    SPLUNK = "splunk"
    ELASTIC = "elastic"
    CUSTOM = "custom"

class SecurityLevel(str, Enum):
    """Security levels for hook validation."""
    NONE = "none"          # No security validation
    BASIC = "basic"        # Basic URL and domain validation
    SIGNED = "signed"      # HMAC signature validation
    MUTUAL_TLS = "mutual_tls"  # Mutual TLS authentication

class HookEvent(str, Enum):
    """Types of events that can trigger hooks."""
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    NOTIFICATION = "notification"
    STOP = "stop"
    SUBAGENT_STOP = "subagent_stop"
    SYSTEM_ALERT = "system_alert"
    PERFORMANCE_THRESHOLD = "performance_threshold"
    ERROR_THRESHOLD = "error_threshold"
    SECURITY_EVENT = "security_event"

class ExternalHookConfig(BaseModel):
    """Configuration for an external hook."""
    
    id: str = Field(..., description="Unique hook identifier")
    name: str = Field(..., description="Human-readable hook name")
    hook_type: HookType = Field(..., description="Type of external hook")
    url: HttpUrl = Field(..., description="Target URL for the hook")
    events: List[HookEvent] = Field(..., description="Events that trigger this hook")
    
    # Security configuration
    security_level: SecurityLevel = Field(default=SecurityLevel.BASIC, description="Security validation level")
    secret_key: Optional[str] = Field(None, description="Secret key for HMAC signing")
    allowed_domains: List[str] = Field(default=[], description="Allowed domains for webhook URLs")
    
    # Request configuration
    timeout_seconds: int = Field(default=30, description="Request timeout in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    retry_delay_seconds: int = Field(default=5, description="Delay between retries")
    
    # Rate limiting
    rate_limit_per_minute: int = Field(default=60, description="Maximum requests per minute")
    
    # Filtering
    agent_filter: Optional[List[str]] = Field(None, description="Filter by specific agent IDs")
    session_filter: Optional[List[str]] = Field(None, description="Filter by specific session IDs")
    tool_filter: Optional[List[str]] = Field(None, description="Filter by specific tool names")
    
    # Additional headers
    headers: Dict[str, str] = Field(default={}, description="Additional HTTP headers")
    
    # Configuration
    enabled: bool = Field(default=True, description="Whether this hook is enabled")
    batch_size: int = Field(default=1, description="Number of events to batch together")
    batch_timeout_seconds: int = Field(default=10, description="Maximum time to wait for batch")
    
    @validator('url')
    def validate_url(cls, v):
        """Validate webhook URL."""
        parsed = urlparse(str(v))
        
        # Ensure HTTPS for security
        if parsed.scheme != 'https':
            raise ValueError("Webhook URLs must use HTTPS")
        
        # Block localhost and private IPs for security
        if parsed.hostname in ['localhost', '127.0.0.1', '0.0.0.0']:
            raise ValueError("Localhost URLs are not allowed")
        
        # Block private IP ranges
        import ipaddress
        try:
            ip = ipaddress.ip_address(parsed.hostname)
            if ip.is_private:
                raise ValueError("Private IP addresses are not allowed")
        except ValueError:
            # Not an IP address, proceed with domain validation
            pass
        
        return v
    
    @validator('allowed_domains')
    def validate_domains(cls, v, values):
        """Validate allowed domains against URL."""
        if not v:
            return v
        
        url = values.get('url')
        if url:
            parsed = urlparse(str(url))
            if parsed.hostname not in v:
                raise ValueError(f"URL domain {parsed.hostname} not in allowed domains")
        
        return v

class RateLimiter:
    """Rate limiter for external hooks."""
    
    def __init__(self):
        self.requests: Dict[str, List[float]] = defaultdict(list)
    
    def is_allowed(self, hook_id: str, limit_per_minute: int) -> bool:
        """Check if request is allowed under rate limit."""
        now = time.time()
        window_start = now - 60  # 1 minute window
        
        # Clean old requests
        self.requests[hook_id] = [
            req_time for req_time in self.requests[hook_id]
            if req_time > window_start
        ]
        
        # Check limit
        if len(self.requests[hook_id]) >= limit_per_minute:
            return False
        
        # Record this request
        self.requests[hook_id].append(now)
        return True

class HookAuditLog(BaseModel):
    """Audit log entry for hook execution."""
    
    hook_id: str
    hook_name: str
    event_type: HookEvent
    timestamp: datetime
    request_url: str
    request_headers: Dict[str, str]
    request_payload: Dict[str, Any]
    response_status: Optional[int] = None
    response_headers: Optional[Dict[str, str]] = None
    response_body: Optional[str] = None
    execution_time_ms: Optional[int] = None
    success: bool
    error_message: Optional[str] = None
    security_validated: bool
    rate_limited: bool

class ExternalHookManager:
    """
    Manager for external hook integrations with security validation and audit trails.
    
    Provides secure webhook delivery, rate limiting, retry logic, and comprehensive
    audit logging for compliance and debugging.
    """
    
    def __init__(self, hook_interceptor: HookInterceptor):
        """Initialize the external hook manager."""
        self.hook_interceptor = hook_interceptor
        self.hooks: Dict[str, ExternalHookConfig] = {}
        self.rate_limiter = RateLimiter()
        self.audit_logs: List[HookAuditLog] = []
        self.max_audit_logs = 10000  # Keep last 10k audit entries
        
        # Batch processing
        self.batch_queues: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.batch_timers: Dict[str, asyncio.Task] = {}
        
        # Security settings
        self.blocked_domains: Set[str] = {
            'localhost', '127.0.0.1', '0.0.0.0', '::1',
            '10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16'
        }
        
        logger.info("🔗 External Hook Manager initialized")
    
    def register_hook(self, config: ExternalHookConfig) -> None:
        """Register a new external hook."""
        try:
            # Validate configuration
            self._validate_hook_config(config)
            
            # Store configuration
            self.hooks[config.id] = config
            
            logger.info(
                "✅ External hook registered",
                hook_id=config.id,
                hook_name=config.name,
                hook_type=config.hook_type,
                url=str(config.url),
                events=config.events,
                security_level=config.security_level
            )
            
        except Exception as e:
            logger.error(
                "❌ Failed to register external hook",
                hook_id=config.id,
                error=str(e)
            )
            raise
    
    def unregister_hook(self, hook_id: str) -> None:
        """Unregister an external hook."""
        if hook_id in self.hooks:
            config = self.hooks.pop(hook_id)
            
            # Cancel any pending batch timer
            if hook_id in self.batch_timers:
                self.batch_timers[hook_id].cancel()
                del self.batch_timers[hook_id]
            
            # Clear batch queue
            if hook_id in self.batch_queues:
                del self.batch_queues[hook_id]
            
            logger.info(
                "🗑️ External hook unregistered",
                hook_id=hook_id,
                hook_name=config.name
            )
        else:
            logger.warning("⚠️ Attempted to unregister unknown hook", hook_id=hook_id)
    
    def get_hook(self, hook_id: str) -> Optional[ExternalHookConfig]:
        """Get hook configuration by ID."""
        return self.hooks.get(hook_id)
    
    def list_hooks(self) -> List[ExternalHookConfig]:
        """List all registered hooks."""
        return list(self.hooks.values())
    
    async def trigger_hooks(
        self,
        event_type: HookEvent,
        payload: Dict[str, Any],
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        tool_name: Optional[str] = None
    ) -> List[HookAuditLog]:
        """
        Trigger all matching external hooks for an event.
        
        Args:
            event_type: Type of event
            payload: Event payload data
            session_id: Optional session ID for filtering
            agent_id: Optional agent ID for filtering
            tool_name: Optional tool name for filtering
            
        Returns:
            List of audit log entries for executed hooks
        """
        audit_logs = []
        
        # Find matching hooks
        matching_hooks = [
            config for config in self.hooks.values()
            if self._should_trigger_hook(config, event_type, session_id, agent_id, tool_name)
        ]
        
        if not matching_hooks:
            return audit_logs
        
        logger.debug(
            "🎯 Triggering external hooks",
            event_type=event_type,
            matching_hooks=len(matching_hooks)
        )
        
        # Execute hooks
        tasks = []
        for config in matching_hooks:
            if config.batch_size > 1:
                # Use batch processing
                task = self._queue_for_batch(config, event_type, payload, session_id, agent_id)
            else:
                # Execute immediately
                task = self._execute_hook(config, event_type, payload, session_id, agent_id)
            
            tasks.append(task)
        
        # Wait for all hooks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect audit logs
        for i, result in enumerate(results):
            if isinstance(result, HookAuditLog):
                audit_logs.append(result)
            elif isinstance(result, Exception):
                logger.error(
                    "❌ Hook execution failed",
                    hook_id=matching_hooks[i].id,
                    error=str(result)
                )
        
        return audit_logs
    
    def _validate_hook_config(self, config: ExternalHookConfig) -> None:
        """Validate hook configuration for security and correctness."""
        # URL security validation
        parsed_url = urlparse(str(config.url))
        
        # Check domain allowlist if specified
        if config.allowed_domains and parsed_url.hostname not in config.allowed_domains:
            raise ValueError(f"Domain {parsed_url.hostname} not in allowed domains")
        
        # Check against blocked domains
        if parsed_url.hostname in self.blocked_domains:
            raise ValueError(f"Domain {parsed_url.hostname} is blocked")
        
        # Validate security configuration
        if config.security_level == SecurityLevel.SIGNED and not config.secret_key:
            raise ValueError("Secret key required for signed security level")
        
        # Validate rate limiting
        if config.rate_limit_per_minute <= 0:
            raise ValueError("Rate limit must be positive")
        
        # Validate batch configuration
        if config.batch_size < 1:
            raise ValueError("Batch size must be at least 1")
    
    def _should_trigger_hook(
        self,
        config: ExternalHookConfig,
        event_type: HookEvent,
        session_id: Optional[str],
        agent_id: Optional[str],
        tool_name: Optional[str]
    ) -> bool:
        """Check if hook should be triggered for this event."""
        # Check if hook is enabled
        if not config.enabled:
            return False
        
        # Check event type
        if event_type not in config.events:
            return False
        
        # Check agent filter
        if config.agent_filter and agent_id not in config.agent_filter:
            return False
        
        # Check session filter
        if config.session_filter and session_id not in config.session_filter:
            return False
        
        # Check tool filter
        if config.tool_filter and tool_name not in config.tool_filter:
            return False
        
        # Check rate limiting
        if not self.rate_limiter.is_allowed(config.id, config.rate_limit_per_minute):
            logger.warning(
                "⏱️ Hook rate limited",
                hook_id=config.id,
                rate_limit=config.rate_limit_per_minute
            )
            return False
        
        return True
    
    async def _queue_for_batch(
        self,
        config: ExternalHookConfig,
        event_type: HookEvent,
        payload: Dict[str, Any],
        session_id: Optional[str],
        agent_id: Optional[str]
    ) -> Optional[HookAuditLog]:
        """Queue event for batch processing."""
        event_data = {
            "event_type": event_type,
            "payload": payload,
            "session_id": session_id,
            "agent_id": agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Add to batch queue
        self.batch_queues[config.id].append(event_data)
        
        # Check if batch is full
        if len(self.batch_queues[config.id]) >= config.batch_size:
            return await self._execute_batch(config)
        
        # Set timer if not already set
        if config.id not in self.batch_timers:
            self.batch_timers[config.id] = asyncio.create_task(
                self._batch_timeout(config)
            )
        
        return None
    
    async def _batch_timeout(self, config: ExternalHookConfig) -> None:
        """Handle batch timeout."""
        await asyncio.sleep(config.batch_timeout_seconds)
        
        if config.id in self.batch_queues and self.batch_queues[config.id]:
            await self._execute_batch(config)
        
        # Clean up timer
        if config.id in self.batch_timers:
            del self.batch_timers[config.id]
    
    async def _execute_batch(self, config: ExternalHookConfig) -> HookAuditLog:
        """Execute a batch of events."""
        batch_events = self.batch_queues[config.id].copy()
        self.batch_queues[config.id].clear()
        
        # Cancel timer if exists
        if config.id in self.batch_timers:
            self.batch_timers[config.id].cancel()
            del self.batch_timers[config.id]
        
        # Create batch payload
        batch_payload = {
            "batch": True,
            "events": batch_events,
            "batch_size": len(batch_events),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self._execute_hook(
            config, 
            HookEvent.NOTIFICATION,  # Use generic event type for batches
            batch_payload,
            None,
            None
        )
    
    async def _execute_hook(
        self,
        config: ExternalHookConfig,
        event_type: HookEvent,
        payload: Dict[str, Any],
        session_id: Optional[str],
        agent_id: Optional[str]
    ) -> HookAuditLog:
        """Execute a single hook with security validation and audit logging."""
        start_time = time.time()
        
        # Prepare request payload
        request_payload = {
            "hook_id": config.id,
            "hook_type": config.hook_type,
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "agent_id": agent_id,
            "payload": payload
        }
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "LeanVibe-Agent-Hive/2.0",
            **config.headers
        }
        
        # Add signature for signed security level
        security_validated = False
        if config.security_level == SecurityLevel.SIGNED and config.secret_key:
            signature = self._generate_signature(request_payload, config.secret_key)
            headers["X-Hub-Signature-256"] = f"sha256={signature}"
            security_validated = True
        elif config.security_level == SecurityLevel.BASIC:
            security_validated = True
        
        # Initialize audit log
        audit_log = HookAuditLog(
            hook_id=config.id,
            hook_name=config.name,
            event_type=event_type,
            timestamp=datetime.utcnow(),
            request_url=str(config.url),
            request_headers=headers,
            request_payload=request_payload,
            success=False,
            security_validated=security_validated,
            rate_limited=False
        )
        
        # Execute request with retries
        for attempt in range(config.retry_attempts):
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=config.timeout_seconds)
                ) as session:
                    async with session.post(
                        str(config.url),
                        json=request_payload,
                        headers=headers
                    ) as response:
                        execution_time_ms = int((time.time() - start_time) * 1000)
                        
                        # Update audit log with response
                        audit_log.response_status = response.status
                        audit_log.response_headers = dict(response.headers)
                        audit_log.execution_time_ms = execution_time_ms
                        audit_log.success = 200 <= response.status < 300
                        
                        # Read response body (limited)
                        try:
                            response_text = await response.text()
                            audit_log.response_body = response_text[:1000]  # Limit size
                        except Exception:
                            audit_log.response_body = "<failed to read response>"
                        
                        if audit_log.success:
                            logger.debug(
                                "✅ Hook executed successfully",
                                hook_id=config.id,
                                status=response.status,
                                execution_time_ms=execution_time_ms
                            )
                            break
                        else:
                            logger.warning(
                                "⚠️ Hook returned error status",
                                hook_id=config.id,
                                status=response.status,
                                attempt=attempt + 1
                            )
                            
                            if attempt < config.retry_attempts - 1:
                                await asyncio.sleep(config.retry_delay_seconds)
                        
            except Exception as e:
                execution_time_ms = int((time.time() - start_time) * 1000)
                audit_log.execution_time_ms = execution_time_ms
                audit_log.error_message = str(e)
                
                logger.error(
                    "❌ Hook execution failed",
                    hook_id=config.id,
                    attempt=attempt + 1,
                    error=str(e)
                )
                
                if attempt < config.retry_attempts - 1:
                    await asyncio.sleep(config.retry_delay_seconds)
        
        # Store audit log
        self._store_audit_log(audit_log)
        
        return audit_log
    
    def _generate_signature(self, payload: Dict[str, Any], secret_key: str) -> str:
        """Generate HMAC signature for webhook payload."""
        payload_json = json.dumps(payload, sort_keys=True)
        signature = hmac.new(
            secret_key.encode(),
            payload_json.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _store_audit_log(self, audit_log: HookAuditLog) -> None:
        """Store audit log entry."""
        self.audit_logs.append(audit_log)
        
        # Trim audit logs if necessary
        if len(self.audit_logs) > self.max_audit_logs:
            self.audit_logs = self.audit_logs[-self.max_audit_logs:]
    
    def get_audit_logs(
        self,
        hook_id: Optional[str] = None,
        event_type: Optional[HookEvent] = None,
        success: Optional[bool] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[HookAuditLog]:
        """Get audit logs with optional filtering."""
        logs = self.audit_logs
        
        # Apply filters
        if hook_id:
            logs = [log for log in logs if log.hook_id == hook_id]
        
        if event_type:
            logs = [log for log in logs if log.event_type == event_type]
        
        if success is not None:
            logs = [log for log in logs if log.success == success]
        
        if since:
            logs = [log for log in logs if log.timestamp >= since]
        
        # Sort by timestamp (newest first) and limit
        logs.sort(key=lambda x: x.timestamp, reverse=True)
        return logs[:limit]
    
    def get_hook_statistics(self, hook_id: str) -> Dict[str, Any]:
        """Get statistics for a specific hook."""
        hook_logs = [log for log in self.audit_logs if log.hook_id == hook_id]
        
        if not hook_logs:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "average_execution_time_ms": 0.0,
                "last_execution": None
            }
        
        total = len(hook_logs)
        successful = sum(1 for log in hook_logs if log.success)
        
        execution_times = [
            log.execution_time_ms for log in hook_logs 
            if log.execution_time_ms is not None
        ]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        return {
            "total_executions": total,
            "success_rate": successful / total if total > 0 else 0.0,
            "average_execution_time_ms": avg_execution_time,
            "last_execution": max(log.timestamp for log in hook_logs),
            "recent_errors": [
                {"timestamp": log.timestamp, "error": log.error_message}
                for log in hook_logs[-10:] 
                if not log.success and log.error_message
            ]
        }

# Global external hook manager
_external_hook_manager: Optional[ExternalHookManager] = None

def get_external_hook_manager() -> Optional[ExternalHookManager]:
    """Get the global external hook manager."""
    return _external_hook_manager

def set_external_hook_manager(manager: ExternalHookManager) -> None:
    """Set the global external hook manager."""
    global _external_hook_manager
    _external_hook_manager = manager
    logger.info("🔗 Global external hook manager set")