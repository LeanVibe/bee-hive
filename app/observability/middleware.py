"""
Observability middleware for LeanVibe Agent Hive 2.0

Provides comprehensive monitoring, metrics collection, event tracking,
and hook interception for multi-agent coordination workflows.
"""

import re
import time
import uuid
from typing import Callable, Any, Optional

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.observability.hooks import get_hook_integration_manager

try:
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    # Mock prometheus metrics if not available
    class MockMetric:
        def labels(self, **kwargs):
            return self
        def inc(self, amount=1):
            pass
        def observe(self, amount):
            pass
        def set(self, value):
            pass
    
    Counter = lambda *args, **kwargs: MockMetric()
    Histogram = lambda *args, **kwargs: MockMetric()
    Gauge = lambda *args, **kwargs: MockMetric()
    PROMETHEUS_AVAILABLE = False

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

ACTIVE_AGENTS = Gauge(
    'active_agents_total',
    'Number of active agents'
)

ACTIVE_SESSIONS = Gauge(
    'active_sessions_total', 
    'Number of active sessions'
)

TASKS_IN_PROGRESS = Gauge(
    'tasks_in_progress_total',
    'Number of tasks currently in progress'
)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """
    Middleware for comprehensive observability of agent operations.
    
    Tracks request metrics, response times, and provides correlation IDs
    for distributed tracing across agent interactions.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with observability tracking."""
        
        # Generate correlation ID for request tracking
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        
        # Extract request information
        method = request.method
        path = request.url.path
        endpoint = self._normalize_endpoint(path)
        
        # Start timing
        start_time = time.time()
        
        # Add structured logging context
        logger = structlog.get_logger().bind(
            correlation_id=correlation_id,
            method=method,
            endpoint=endpoint,
            user_agent=request.headers.get("user-agent", "unknown")
        )
        
        logger.info("Request started")
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            status_code = response.status_code
            
            # Record metrics
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()
            
            REQUEST_DURATION.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            
            # Log request completion
            logger.info(
                "Request completed",
                status_code=status_code,
                duration=duration,
                response_size=response.headers.get("content-length", "unknown")
            )
            
            return response
            
        except Exception as e:
            # Calculate duration for failed requests
            duration = time.time() - start_time
            
            # Record failed request metrics
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status_code=500
            ).inc()
            
            REQUEST_DURATION.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            # Log error
            logger.error(
                "Request failed",
                error=str(e),
                duration=duration,
                exc_info=True
            )
            
            raise
    
    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint path for metrics aggregation."""
        
        # Remove UUID parameters to avoid high cardinality
        import re
        
        # Replace UUIDs with placeholder
        uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        normalized = re.sub(uuid_pattern, '{id}', path, flags=re.IGNORECASE)
        
        # Remove query parameters
        if '?' in normalized:
            normalized = normalized.split('?')[0]
        
        return normalized


class AgentEventTracker:
    """Track agent-specific events for observability."""
    
    def __init__(self):
        self.agent_events = Counter(
            'agent_events_total',
            'Total agent events',
            ['agent_id', 'event_type', 'status']
        )
        
        self.task_events = Counter(
            'task_events_total',
            'Total task events', 
            ['task_type', 'event_type', 'status']
        )
        
        self.context_operations = Counter(
            'context_operations_total',
            'Total context operations',
            ['operation_type', 'context_type']
        )
    
    def track_agent_event(
        self,
        agent_id: str,
        event_type: str,
        status: str = "success",
        **kwargs: Any
    ) -> None:
        """Track an agent-specific event."""
        
        self.agent_events.labels(
            agent_id=agent_id,
            event_type=event_type,
            status=status
        ).inc()
        
        logger.info(
            "Agent event tracked",
            agent_id=agent_id,
            event_type=event_type,
            status=status,
            **kwargs
        )
    
    def track_task_event(
        self,
        task_type: str,
        event_type: str,
        status: str = "success",
        **kwargs: Any
    ) -> None:
        """Track a task-specific event."""
        
        self.task_events.labels(
            task_type=task_type,
            event_type=event_type,
            status=status
        ).inc()
        
        logger.info(
            "Task event tracked",
            task_type=task_type,
            event_type=event_type,
            status=status,
            **kwargs
        )
    
    def track_context_operation(
        self,
        operation_type: str,
        context_type: str,
        **kwargs: Any
    ) -> None:
        """Track a context operation."""
        
        self.context_operations.labels(
            operation_type=operation_type,
            context_type=context_type
        ).inc()
        
        logger.info(
            "Context operation tracked",
            operation_type=operation_type,
            context_type=context_type,
            **kwargs
        )


# Global event tracker instance
event_tracker = AgentEventTracker()


class ObservabilityHookMiddleware(BaseHTTPMiddleware):
    """
    Middleware for capturing agent tool execution events automatically.
    
    Intercepts API calls to agent endpoints and captures PreToolUse/PostToolUse
    events for comprehensive observability and monitoring.
    """
    
    def __init__(self, app, capture_patterns: Optional[list] = None):
        """
        Initialize ObservabilityHookMiddleware.
        
        Args:
            app: FastAPI application instance
            capture_patterns: List of regex patterns for endpoints to capture
        """
        super().__init__(app)
        
        # Default patterns for agent tool execution endpoints
        self.capture_patterns = capture_patterns or [
            r"/agents/[^/]+/tools/execute",
            r"/agents/[^/]+/action",
            r"/api/v1/agents/[^/]+/tools/execute",
            r"/api/v1/code_execution/execute",
            r"/api/v1/external_tools/execute"
        ]
        
        # Compile regex patterns
        self.compiled_patterns = [re.compile(pattern) for pattern in self.capture_patterns]
        
        logger.info(
            "🔗 ObservabilityHookMiddleware initialized",
            capture_patterns=self.capture_patterns
        )
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with automatic hook capture."""
        
        # Check if this endpoint should be captured
        should_capture = self._should_capture_endpoint(request.url.path)
        
        if not should_capture:
            # No capture needed, proceed normally
            return await call_next(request)
        
        # Get hook interceptor
        interceptor = get_hook_integration_manager()
        if not interceptor or not interceptor.is_enabled:
            # Hook interceptor not available or disabled
            return await call_next(request)
        
        # Extract agent and session info from request
        agent_id, session_id = await self._extract_agent_session_info(request)
        
        if not agent_id or not session_id:
            # Cannot extract required info, proceed without capture
            return await call_next(request)
        
        # Capture PreToolUse event
        correlation_id = getattr(request.state, 'correlation_id', str(uuid.uuid4()))
        pre_event_id = None
        
        try:
            # Get tool data from request
            tool_data = await self._extract_tool_data(request)
            tool_data["correlation_id"] = correlation_id
            
            pre_event_id = await interceptor.capture_pre_tool_use(
                session_id=session_id,
                agent_id=agent_id,
                tool_data=tool_data
            )
            
        except Exception as e:
            logger.warning(
                "⚠️ Failed to capture PreToolUse event",
                error=str(e),
                path=request.url.path
            )
        
        # Execute the actual request
        start_time = time.time()
        response = await call_next(request)
        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)
        
        # Capture PostToolUse event
        try:
            tool_result = await self._extract_tool_result(request, response, latency_ms)
            tool_result["correlation_id"] = correlation_id
            
            await interceptor.capture_post_tool_use(
                session_id=session_id,
                agent_id=agent_id,
                tool_result=tool_result,
                latency_ms=latency_ms
            )
            
        except Exception as e:
            logger.warning(
                "⚠️ Failed to capture PostToolUse event",
                error=str(e),
                path=request.url.path
            )
        
        # Add observability headers
        if pre_event_id:
            response.headers["X-Observability-Events-Captured"] = "true"
            response.headers["X-Pre-Event-ID"] = pre_event_id
        
        response.headers["X-Correlation-ID"] = correlation_id
        
        return response
    
    def _should_capture_endpoint(self, path: str) -> bool:
        """Check if endpoint should be captured based on patterns."""
        for pattern in self.compiled_patterns:
            if pattern.search(path):
                return True
        return False
    
    async def _extract_agent_session_info(self, request: Request) -> tuple[Optional[uuid.UUID], Optional[uuid.UUID]]:
        """Extract agent ID and session ID from request."""
        try:
            # Try to extract from URL path
            path = request.url.path
            
            # Pattern: /agents/{agent_id}/...
            agent_match = re.search(r"/agents/([^/]+)", path)
            if agent_match:
                agent_id = uuid.UUID(agent_match.group(1))
            else:
                # Try to extract from headers
                agent_id_str = request.headers.get("X-Agent-ID")
                agent_id = uuid.UUID(agent_id_str) if agent_id_str else None
            
            # Try to extract session ID from headers or query params
            session_id_str = (
                request.headers.get("X-Session-ID") or
                request.query_params.get("session_id")
            )
            session_id = uuid.UUID(session_id_str) if session_id_str else None
            
            # If no session ID found, generate one
            if not session_id:
                session_id = uuid.uuid4()
                logger.debug("Generated session ID for request", session_id=str(session_id))
            
            return agent_id, session_id
            
        except Exception as e:
            logger.warning("Failed to extract agent/session info", error=str(e))
            return None, None
    
    async def _extract_tool_data(self, request: Request) -> dict:
        """Extract tool execution data from request."""
        try:
            # Determine tool name from endpoint
            path = request.url.path
            tool_name = "unknown"
            
            if "code_execution" in path:
                tool_name = "CodeExecution"
            elif "external_tools" in path:
                tool_name = "ExternalTool"
            elif "tools/execute" in path:
                tool_name = "ToolExecution"
            
            # Try to get request body for parameters
            parameters = {}
            
            # Clone request body safely
            if hasattr(request, "_body"):
                try:
                    import json
                    body_str = request._body.decode() if request._body else "{}"
                    body_data = json.loads(body_str)
                    
                    # Extract relevant parameters, sanitizing sensitive data
                    if isinstance(body_data, dict):
                        parameters = {
                            k: v for k, v in body_data.items()
                            if k not in ['password', 'token', 'secret', 'key']
                        }
                        
                        # Limit parameter size to prevent payload bloat
                        if len(str(parameters)) > 5000:
                            parameters = {"_truncated": True, "_size": len(str(parameters))}
                    
                except Exception:
                    parameters = {"_parse_error": True}
            
            return {
                "tool_name": tool_name,
                "parameters": parameters,
                "endpoint": path,
                "method": request.method
            }
            
        except Exception as e:
            logger.warning("Failed to extract tool data", error=str(e))
            return {"tool_name": "unknown", "parameters": {}}
    
    async def _extract_tool_result(self, request: Request, response: Response, latency_ms: int) -> dict:
        """Extract tool execution result from response."""
        try:
            # Determine success based on status code
            success = 200 <= response.status_code < 300
            
            # Extract tool name (same logic as pre-tool)
            path = request.url.path
            tool_name = "unknown"
            
            if "code_execution" in path:
                tool_name = "CodeExecution"
            elif "external_tools" in path:
                tool_name = "ExternalTool"
            elif "tools/execute" in path:
                tool_name = "ToolExecution"
            
            result_data = {
                "tool_name": tool_name,
                "success": success,
                "status_code": response.status_code,
                "execution_time_ms": latency_ms
            }
            
            # Try to extract error information for failed requests
            if not success:
                # Add error details based on status code
                if response.status_code == 422:
                    result_data["error"] = "Validation error"
                    result_data["error_type"] = "ValidationError"
                elif response.status_code == 500:
                    result_data["error"] = "Internal server error"
                    result_data["error_type"] = "InternalServerError"
                elif response.status_code == 404:
                    result_data["error"] = "Not found"
                    result_data["error_type"] = "NotFoundError"
                else:
                    result_data["error"] = f"HTTP {response.status_code}"
                    result_data["error_type"] = "HTTPError"
            
            return result_data
            
        except Exception as e:
            logger.warning("Failed to extract tool result", error=str(e))
            return {
                "tool_name": "unknown",
                "success": False,
                "error": "Failed to extract result",
                "error_type": "ExtractionError"
            }