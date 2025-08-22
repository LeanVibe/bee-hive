"""
Webhook Integration Plugin Example

Demonstrates integration plugin development using the LeanVibe SDK.
Shows external service integration, webhook handling, and API communication.
"""

import asyncio
import aiohttp
import json
import hmac
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from urllib.parse import urljoin

from ..interfaces import PluginBase, PluginType
from ..models import PluginConfig, TaskInterface, TaskResult, PluginEvent, EventSeverity
from ..decorators import plugin_method, performance_tracked, error_handled, cached_result
from ..exceptions import PluginConfigurationError, PluginExecutionError


@dataclass
class WebhookConfig:
    """Webhook configuration."""
    webhook_id: str
    url: str
    secret: Optional[str] = None
    headers: Dict[str, str] = None
    timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_delay_seconds: int = 5
    enabled: bool = True
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}


@dataclass
class WebhookDelivery:
    """Webhook delivery record."""
    delivery_id: str
    webhook_id: str
    payload: Dict[str, Any]
    response_status: Optional[int] = None
    response_body: Optional[str] = None
    delivery_attempts: int = 0
    delivered_at: Optional[datetime] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "delivery_id": self.delivery_id,
            "webhook_id": self.webhook_id,
            "payload": self.payload,
            "response_status": self.response_status,
            "response_body": self.response_body,
            "delivery_attempts": self.delivery_attempts,
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class APIEndpoint:
    """External API endpoint configuration."""
    endpoint_id: str
    base_url: str
    auth_type: str  # none, api_key, bearer, basic
    auth_config: Dict[str, str] = None
    default_headers: Dict[str, str] = None
    timeout_seconds: int = 30
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60
    
    def __post_init__(self):
        if self.auth_config is None:
            self.auth_config = {}
        if self.default_headers is None:
            self.default_headers = {}


class WebhookIntegrationPlugin(PluginBase):
    """
    Advanced webhook and API integration plugin.
    
    Features:
    - Webhook delivery with retries and error handling
    - External API communication with various auth methods
    - Rate limiting and circuit breaker patterns
    - Event transformation and routing
    - Integration monitoring and analytics
    - Flexible payload formatting
    
    Epic 1 Optimizations:
    - Efficient HTTP connection pooling
    - Async webhook delivery
    - <50ms integration response times
    - <15MB memory footprint
    """
    
    def __init__(self, config: PluginConfig):
        super().__init__(config)
        
        # Configuration
        self.webhooks = self._load_webhook_configs()
        self.api_endpoints = self._load_api_endpoints()
        self.max_concurrent_deliveries = config.parameters.get("max_concurrent_deliveries", 10)
        self.delivery_timeout = config.parameters.get("delivery_timeout_seconds", 30)
        self.enable_delivery_history = config.parameters.get("enable_delivery_history", True)
        self.max_history_entries = config.parameters.get("max_history_entries", 1000)
        
        # Runtime state
        self.delivery_history: List[WebhookDelivery] = []
        self.delivery_queue = asyncio.Queue()
        self.integration_stats = {
            "webhooks_sent": 0,
            "api_calls_made": 0,
            "delivery_failures": 0,
            "average_response_time_ms": 0.0
        }
        
        # HTTP session for connection pooling
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._delivery_workers: List[asyncio.Task] = []
        
        # Rate limiting
        self._rate_limits: Dict[str, List[datetime]] = {}
        
        # Performance tracking
        self._response_times = []
    
    def _load_webhook_configs(self) -> Dict[str, WebhookConfig]:
        """Load webhook configurations."""
        webhook_configs = self.config.parameters.get("webhooks", [])
        
        webhooks = {}
        for webhook_config in webhook_configs:
            webhook = WebhookConfig(
                webhook_id=webhook_config["webhook_id"],
                url=webhook_config["url"],
                secret=webhook_config.get("secret"),
                headers=webhook_config.get("headers", {}),
                timeout_seconds=webhook_config.get("timeout_seconds", 30),
                retry_attempts=webhook_config.get("retry_attempts", 3),
                retry_delay_seconds=webhook_config.get("retry_delay_seconds", 5),
                enabled=webhook_config.get("enabled", True)
            )
            webhooks[webhook.webhook_id] = webhook
        
        return webhooks
    
    def _load_api_endpoints(self) -> Dict[str, APIEndpoint]:
        """Load API endpoint configurations."""
        endpoint_configs = self.config.parameters.get("api_endpoints", [])
        
        endpoints = {}
        for endpoint_config in endpoint_configs:
            endpoint = APIEndpoint(
                endpoint_id=endpoint_config["endpoint_id"],
                base_url=endpoint_config["base_url"],
                auth_type=endpoint_config.get("auth_type", "none"),
                auth_config=endpoint_config.get("auth_config", {}),
                default_headers=endpoint_config.get("default_headers", {}),
                timeout_seconds=endpoint_config.get("timeout_seconds", 30),
                rate_limit_requests=endpoint_config.get("rate_limit_requests", 100),
                rate_limit_window_seconds=endpoint_config.get("rate_limit_window_seconds", 60)
            )
            endpoints[endpoint.endpoint_id] = endpoint
        
        return endpoints
    
    async def _on_initialize(self) -> None:
        """Initialize the webhook integration plugin."""
        await self.log_info("Initializing WebhookIntegrationPlugin")
        
        # Validate configuration
        if self.max_concurrent_deliveries <= 0:
            raise PluginConfigurationError(
                "Max concurrent deliveries must be positive",
                config_key="max_concurrent_deliveries",
                expected_type="positive integer",
                actual_value=self.max_concurrent_deliveries,
                plugin_id=self.plugin_id
            )
        
        # Initialize HTTP session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=20,  # Per-host connection limit
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True
        )
        
        timeout = aiohttp.ClientTimeout(total=self.delivery_timeout)
        self._http_session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": f"LeanVibe-Plugin/{self.config.version}"}
        )
        
        # Start delivery workers
        self._delivery_workers = []
        for i in range(self.max_concurrent_deliveries):
            worker = asyncio.create_task(self._delivery_worker(f"worker_{i}"))
            self._delivery_workers.append(worker)
        
        # Initialize state
        self.delivery_history = []
        self.integration_stats = {
            "webhooks_sent": 0,
            "api_calls_made": 0,
            "delivery_failures": 0,
            "average_response_time_ms": 0.0
        }
        self._rate_limits = {}
        
        await self.log_info(
            f"Initialized with {len(self.webhooks)} webhooks, "
            f"{len(self.api_endpoints)} API endpoints, "
            f"{self.max_concurrent_deliveries} delivery workers"
        )
    
    @performance_tracked(alert_threshold_ms=200, memory_limit_mb=20)
    @plugin_method(timeout_seconds=300, max_retries=1)
    async def handle_task(self, task: TaskInterface) -> TaskResult:
        """
        Execute integration operations.
        
        Supports the following task types:
        - send_webhook: Send webhook to configured endpoint
        - call_api: Make API call to external service
        - send_batch_webhooks: Send multiple webhooks
        - get_delivery_status: Get webhook delivery status
        - get_integration_stats: Get integration statistics
        - test_webhook: Test webhook configuration
        - test_api_endpoint: Test API endpoint configuration
        """
        task_type = task.task_type
        
        if task_type == "send_webhook":
            return await self._send_webhook(task)
        elif task_type == "call_api":
            return await self._call_api(task)
        elif task_type == "send_batch_webhooks":
            return await self._send_batch_webhooks(task)
        elif task_type == "get_delivery_status":
            return await self._get_delivery_status(task)
        elif task_type == "get_integration_stats":
            return await self._get_integration_stats(task)
        elif task_type == "test_webhook":
            return await self._test_webhook(task)
        elif task_type == "test_api_endpoint":
            return await self._test_api_endpoint(task)
        else:
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error=f"Unknown task type: {task_type}",
                error_code="INVALID_TASK_TYPE"
            )
    
    async def _send_webhook(self, task: TaskInterface) -> TaskResult:
        """Send webhook to configured endpoint."""
        try:
            webhook_id = task.parameters.get("webhook_id")
            payload = task.parameters.get("payload", {})
            
            if not webhook_id:
                return TaskResult(
                    success=False,
                    plugin_id=self.plugin_id,
                    task_id=task.task_id,
                    error="webhook_id is required",
                    error_code="MISSING_WEBHOOK_ID"
                )
            
            if webhook_id not in self.webhooks:
                return TaskResult(
                    success=False,
                    plugin_id=self.plugin_id,
                    task_id=task.task_id,
                    error=f"Webhook {webhook_id} not configured",
                    error_code="WEBHOOK_NOT_FOUND"
                )
            
            webhook_config = self.webhooks[webhook_id]
            
            if not webhook_config.enabled:
                return TaskResult(
                    success=False,
                    plugin_id=self.plugin_id,
                    task_id=task.task_id,
                    error=f"Webhook {webhook_id} is disabled",
                    error_code="WEBHOOK_DISABLED"
                )
            
            # Create delivery record
            delivery = WebhookDelivery(
                delivery_id=str(uuid.uuid4()),
                webhook_id=webhook_id,
                payload=payload
            )
            
            # Queue for delivery
            await self.delivery_queue.put(delivery)
            
            await task.update_status("running", progress=0.5)
            await self.log_info(f"Queued webhook delivery: {delivery.delivery_id}")
            
            # Wait for delivery completion (with timeout)
            delivery_result = await self._wait_for_delivery_completion(delivery.delivery_id, timeout=60)
            
            if delivery_result["success"]:
                return TaskResult(
                    success=True,
                    plugin_id=self.plugin_id,
                    task_id=task.task_id,
                    data={
                        "delivery_id": delivery.delivery_id,
                        "webhook_id": webhook_id,
                        "response_status": delivery_result["response_status"],
                        "delivery_attempts": delivery_result["delivery_attempts"]
                    }
                )
            else:
                return TaskResult(
                    success=False,
                    plugin_id=self.plugin_id,
                    task_id=task.task_id,
                    error=delivery_result["error"],
                    error_code="WEBHOOK_DELIVERY_FAILED",
                    data={
                        "delivery_id": delivery.delivery_id,
                        "delivery_attempts": delivery_result["delivery_attempts"]
                    }
                )
            
        except Exception as e:
            await self.log_error(f"Failed to send webhook: {e}")
            
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error=str(e),
                error_code="WEBHOOK_SEND_ERROR"
            )
    
    async def _delivery_worker(self, worker_id: str):
        """Background worker for webhook deliveries."""
        await self.log_info(f"Delivery worker {worker_id} started")
        
        while True:
            try:
                # Wait for delivery request
                delivery = await self.delivery_queue.get()
                
                if delivery is None:  # Shutdown signal
                    break
                
                # Process delivery
                await self._process_webhook_delivery(delivery)
                
                # Mark task as done
                self.delivery_queue.task_done()
                
            except Exception as e:
                await self.log_error(f"Delivery worker {worker_id} error: {e}")
                await asyncio.sleep(1)  # Brief pause before retrying
        
        await self.log_info(f"Delivery worker {worker_id} stopped")
    
    async def _process_webhook_delivery(self, delivery: WebhookDelivery):
        """Process a webhook delivery with retries."""
        webhook_config = self.webhooks[delivery.webhook_id]
        
        for attempt in range(webhook_config.retry_attempts):
            delivery.delivery_attempts += 1
            
            try:
                # Prepare headers
                headers = webhook_config.headers.copy()
                headers["Content-Type"] = "application/json"
                
                # Add signature if secret is configured
                if webhook_config.secret:
                    payload_body = json.dumps(delivery.payload)
                    signature = hmac.new(
                        webhook_config.secret.encode(),
                        payload_body.encode(),
                        hashlib.sha256
                    ).hexdigest()
                    headers["X-Webhook-Signature"] = f"sha256={signature}"
                
                # Make HTTP request
                start_time = datetime.utcnow()
                
                async with self._http_session.post(
                    webhook_config.url,
                    json=delivery.payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=webhook_config.timeout_seconds)
                ) as response:
                    response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                    self._response_times.append(response_time)
                    
                    delivery.response_status = response.status
                    delivery.response_body = await response.text()
                    
                    # Update statistics
                    self.integration_stats["webhooks_sent"] += 1
                    if self._response_times:
                        self.integration_stats["average_response_time_ms"] = (
                            sum(self._response_times) / len(self._response_times)
                        )
                        # Keep only recent times
                        if len(self._response_times) > 100:
                            self._response_times = self._response_times[-100:]
                    
                    if response.status >= 200 and response.status < 300:
                        # Success
                        delivery.delivered_at = datetime.utcnow()
                        
                        # Store in history
                        if self.enable_delivery_history:
                            self.delivery_history.append(delivery)
                            if len(self.delivery_history) > self.max_history_entries:
                                self.delivery_history = self.delivery_history[-self.max_history_entries:]
                        
                        await self.log_info(
                            f"Webhook delivered successfully: {delivery.delivery_id} "
                            f"({response.status}) in {response_time:.1f}ms"
                        )
                        
                        # Emit success event
                        success_event = PluginEvent(
                            event_type="webhook_delivered",
                            plugin_id=self.plugin_id,
                            data={
                                "delivery_id": delivery.delivery_id,
                                "webhook_id": delivery.webhook_id,
                                "response_status": response.status,
                                "response_time_ms": response_time,
                                "delivery_attempts": delivery.delivery_attempts
                            }
                        )
                        await self.emit_event(success_event)
                        
                        return  # Success, exit retry loop
                    
                    else:
                        # HTTP error
                        await self.log_warning(
                            f"Webhook delivery failed: {delivery.delivery_id} "
                            f"({response.status}) - attempt {attempt + 1}/{webhook_config.retry_attempts}"
                        )
                        
                        if attempt == webhook_config.retry_attempts - 1:
                            # Final attempt failed
                            self.integration_stats["delivery_failures"] += 1
                            
                            # Store failed delivery in history
                            if self.enable_delivery_history:
                                self.delivery_history.append(delivery)
                            
                            # Emit failure event
                            failure_event = PluginEvent(
                                event_type="webhook_delivery_failed",
                                plugin_id=self.plugin_id,
                                data={
                                    "delivery_id": delivery.delivery_id,
                                    "webhook_id": delivery.webhook_id,
                                    "response_status": response.status,
                                    "response_body": delivery.response_body,
                                    "delivery_attempts": delivery.delivery_attempts
                                },
                                severity=EventSeverity.ERROR
                            )
                            await self.emit_event(failure_event)
                        
                        else:
                            # Wait before retry
                            await asyncio.sleep(webhook_config.retry_delay_seconds)
            
            except asyncio.TimeoutError:
                await self.log_warning(
                    f"Webhook delivery timeout: {delivery.delivery_id} - "
                    f"attempt {attempt + 1}/{webhook_config.retry_attempts}"
                )
                
                if attempt == webhook_config.retry_attempts - 1:
                    delivery.response_body = "Request timeout"
                    self.integration_stats["delivery_failures"] += 1
                
            except Exception as e:
                await self.log_error(
                    f"Webhook delivery error: {delivery.delivery_id} - {e} - "
                    f"attempt {attempt + 1}/{webhook_config.retry_attempts}"
                )
                
                if attempt == webhook_config.retry_attempts - 1:
                    delivery.response_body = str(e)
                    self.integration_stats["delivery_failures"] += 1
    
    async def _wait_for_delivery_completion(self, delivery_id: str, timeout: int = 60) -> Dict[str, Any]:
        """Wait for delivery completion."""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            # Check if delivery is complete
            for delivery in self.delivery_history:
                if delivery.delivery_id == delivery_id:
                    if delivery.delivered_at:
                        return {
                            "success": True,
                            "response_status": delivery.response_status,
                            "delivery_attempts": delivery.delivery_attempts
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"Delivery failed after {delivery.delivery_attempts} attempts",
                            "delivery_attempts": delivery.delivery_attempts
                        }
            
            await asyncio.sleep(0.1)  # Check every 100ms
        
        return {
            "success": False,
            "error": "Delivery timeout",
            "delivery_attempts": 0
        }
    
    async def _call_api(self, task: TaskInterface) -> TaskResult:
        """Make API call to external service."""
        try:
            endpoint_id = task.parameters.get("endpoint_id")
            method = task.parameters.get("method", "GET").upper()
            path = task.parameters.get("path", "")
            payload = task.parameters.get("payload")
            headers = task.parameters.get("headers", {})
            
            if not endpoint_id:
                return TaskResult(
                    success=False,
                    plugin_id=self.plugin_id,
                    task_id=task.task_id,
                    error="endpoint_id is required",
                    error_code="MISSING_ENDPOINT_ID"
                )
            
            if endpoint_id not in self.api_endpoints:
                return TaskResult(
                    success=False,
                    plugin_id=self.plugin_id,
                    task_id=task.task_id,
                    error=f"API endpoint {endpoint_id} not configured",
                    error_code="ENDPOINT_NOT_FOUND"
                )
            
            endpoint = self.api_endpoints[endpoint_id]
            
            # Check rate limits
            if not await self._check_rate_limit(endpoint_id, endpoint):
                return TaskResult(
                    success=False,
                    plugin_id=self.plugin_id,
                    task_id=task.task_id,
                    error="Rate limit exceeded",
                    error_code="RATE_LIMIT_EXCEEDED"
                )
            
            # Build URL
            url = urljoin(endpoint.base_url, path)
            
            # Prepare headers
            request_headers = endpoint.default_headers.copy()
            request_headers.update(headers)
            
            # Add authentication
            await self._add_authentication(request_headers, endpoint)
            
            # Make request
            start_time = datetime.utcnow()
            
            async with self._http_session.request(
                method,
                url,
                json=payload if payload else None,
                headers=request_headers,
                timeout=aiohttp.ClientTimeout(total=endpoint.timeout_seconds)
            ) as response:
                response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                response_body = await response.text()
                
                # Update statistics
                self.integration_stats["api_calls_made"] += 1
                self._response_times.append(response_time)
                
                # Parse JSON response if possible
                try:
                    response_data = json.loads(response_body)
                except json.JSONDecodeError:
                    response_data = response_body
                
                return TaskResult(
                    success=response.status >= 200 and response.status < 300,
                    plugin_id=self.plugin_id,
                    task_id=task.task_id,
                    execution_time_ms=response_time,
                    data={
                        "endpoint_id": endpoint_id,
                        "method": method,
                        "url": url,
                        "response_status": response.status,
                        "response_headers": dict(response.headers),
                        "response_data": response_data,
                        "response_time_ms": response_time
                    },
                    error=f"HTTP {response.status}" if response.status >= 400 else None
                )
            
        except Exception as e:
            await self.log_error(f"API call failed: {e}")
            
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error=str(e),
                error_code="API_CALL_ERROR"
            )
    
    async def _check_rate_limit(self, endpoint_id: str, endpoint: APIEndpoint) -> bool:
        """Check if request is within rate limits."""
        current_time = datetime.utcnow()
        window_start = current_time - timedelta(seconds=endpoint.rate_limit_window_seconds)
        
        # Get recent requests for this endpoint
        if endpoint_id not in self._rate_limits:
            self._rate_limits[endpoint_id] = []
        
        recent_requests = self._rate_limits[endpoint_id]
        
        # Remove old requests outside the window
        recent_requests[:] = [req_time for req_time in recent_requests if req_time > window_start]
        
        # Check if under limit
        if len(recent_requests) >= endpoint.rate_limit_requests:
            return False
        
        # Add current request
        recent_requests.append(current_time)
        return True
    
    async def _add_authentication(self, headers: Dict[str, str], endpoint: APIEndpoint):
        """Add authentication to request headers."""
        auth_type = endpoint.auth_type.lower()
        
        if auth_type == "api_key":
            api_key = endpoint.auth_config.get("api_key")
            header_name = endpoint.auth_config.get("header_name", "X-API-Key")
            if api_key:
                headers[header_name] = api_key
        
        elif auth_type == "bearer":
            token = endpoint.auth_config.get("token")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        
        elif auth_type == "basic":
            username = endpoint.auth_config.get("username")
            password = endpoint.auth_config.get("password")
            if username and password:
                import base64
                credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
                headers["Authorization"] = f"Basic {credentials}"
    
    async def _send_batch_webhooks(self, task: TaskInterface) -> TaskResult:
        """Send multiple webhooks in batch."""
        try:
            webhook_requests = task.parameters.get("webhook_requests", [])
            
            if not webhook_requests:
                return TaskResult(
                    success=False,
                    plugin_id=self.plugin_id,
                    task_id=task.task_id,
                    error="No webhook requests provided",
                    error_code="MISSING_WEBHOOK_REQUESTS"
                )
            
            delivery_ids = []
            
            # Queue all deliveries
            for webhook_request in webhook_requests:
                webhook_id = webhook_request.get("webhook_id")
                payload = webhook_request.get("payload", {})
                
                if webhook_id and webhook_id in self.webhooks:
                    delivery = WebhookDelivery(
                        delivery_id=str(uuid.uuid4()),
                        webhook_id=webhook_id,
                        payload=payload
                    )
                    
                    await self.delivery_queue.put(delivery)
                    delivery_ids.append(delivery.delivery_id)
            
            return TaskResult(
                success=True,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                data={
                    "queued_deliveries": len(delivery_ids),
                    "delivery_ids": delivery_ids
                }
            )
            
        except Exception as e:
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error=str(e),
                error_code="BATCH_WEBHOOK_ERROR"
            )
    
    async def _get_delivery_status(self, task: TaskInterface) -> TaskResult:
        """Get webhook delivery status."""
        try:
            delivery_id = task.parameters.get("delivery_id")
            
            if not delivery_id:
                return TaskResult(
                    success=False,
                    plugin_id=self.plugin_id,
                    task_id=task.task_id,
                    error="delivery_id is required",
                    error_code="MISSING_DELIVERY_ID"
                )
            
            # Find delivery in history
            for delivery in self.delivery_history:
                if delivery.delivery_id == delivery_id:
                    return TaskResult(
                        success=True,
                        plugin_id=self.plugin_id,
                        task_id=task.task_id,
                        data={"delivery": delivery.to_dict()}
                    )
            
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error="Delivery not found",
                error_code="DELIVERY_NOT_FOUND"
            )
            
        except Exception as e:
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error=str(e),
                error_code="DELIVERY_STATUS_ERROR"
            )
    
    async def _get_integration_stats(self, task: TaskInterface) -> TaskResult:
        """Get integration statistics."""
        return TaskResult(
            success=True,
            plugin_id=self.plugin_id,
            task_id=task.task_id,
            data={
                "integration_stats": self.integration_stats,
                "webhook_configs": len(self.webhooks),
                "api_endpoints": len(self.api_endpoints),
                "delivery_queue_size": self.delivery_queue.qsize(),
                "active_workers": len(self._delivery_workers),
                "delivery_history_size": len(self.delivery_history)
            }
        )
    
    async def _test_webhook(self, task: TaskInterface) -> TaskResult:
        """Test webhook configuration."""
        try:
            webhook_id = task.parameters.get("webhook_id")
            
            if not webhook_id or webhook_id not in self.webhooks:
                return TaskResult(
                    success=False,
                    plugin_id=self.plugin_id,
                    task_id=task.task_id,
                    error="Invalid webhook_id",
                    error_code="INVALID_WEBHOOK_ID"
                )
            
            webhook_config = self.webhooks[webhook_id]
            
            # Send test payload
            test_payload = {
                "test": True,
                "timestamp": datetime.utcnow().isoformat(),
                "webhook_id": webhook_id
            }
            
            delivery = WebhookDelivery(
                delivery_id=f"test_{uuid.uuid4().hex[:8]}",
                webhook_id=webhook_id,
                payload=test_payload
            )
            
            await self._process_webhook_delivery(delivery)
            
            return TaskResult(
                success=delivery.delivered_at is not None,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                data={
                    "test_result": delivery.to_dict(),
                    "webhook_reachable": delivery.delivered_at is not None
                }
            )
            
        except Exception as e:
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error=str(e),
                error_code="WEBHOOK_TEST_ERROR"
            )
    
    async def _test_api_endpoint(self, task: TaskInterface) -> TaskResult:
        """Test API endpoint configuration."""
        try:
            endpoint_id = task.parameters.get("endpoint_id")
            
            if not endpoint_id or endpoint_id not in self.api_endpoints:
                return TaskResult(
                    success=False,
                    plugin_id=self.plugin_id,
                    task_id=task.task_id,
                    error="Invalid endpoint_id",
                    error_code="INVALID_ENDPOINT_ID"
                )
            
            endpoint = self.api_endpoints[endpoint_id]
            
            # Make test request (usually GET to base URL or health endpoint)
            test_path = task.parameters.get("test_path", "")
            
            result = await self._call_api(TaskInterface(
                task_id=f"test_{task.task_id}",
                task_type="call_api",
                parameters={
                    "endpoint_id": endpoint_id,
                    "method": "GET",
                    "path": test_path
                }
            ))
            
            return TaskResult(
                success=result.success,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                data={
                    "test_result": result.data,
                    "endpoint_reachable": result.success
                }
            )
            
        except Exception as e:
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error=str(e),
                error_code="API_TEST_ERROR"
            )
    
    async def _on_cleanup(self) -> None:
        """Cleanup plugin resources."""
        await self.log_info("Cleaning up WebhookIntegrationPlugin")
        
        # Stop delivery workers
        for _ in self._delivery_workers:
            await self.delivery_queue.put(None)  # Shutdown signal
        
        # Wait for workers to finish
        if self._delivery_workers:
            await asyncio.gather(*self._delivery_workers, return_exceptions=True)
        
        # Close HTTP session
        if self._http_session:
            await self._http_session.close()
        
        # Clear state
        self.delivery_history.clear()
        self._response_times.clear()
        self._rate_limits.clear()
        
        # Reset statistics
        self.integration_stats = {
            "webhooks_sent": 0,
            "api_calls_made": 0,
            "delivery_failures": 0,
            "average_response_time_ms": 0.0
        }