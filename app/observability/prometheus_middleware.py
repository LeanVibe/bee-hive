"""
Prometheus Metrics Middleware for HTTP Request Tracking

Automatically tracks HTTP requests and integrates with the Prometheus exporter
to provide comprehensive observability for the LeanVibe Agent Hive API.
"""

import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

import structlog
from ..core.prometheus_exporter import get_prometheus_exporter

logger = structlog.get_logger()

class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Middleware to automatically track HTTP requests in Prometheus metrics.
    
    Tracks:
    - Request count by endpoint, method, and status code
    - Request duration histograms
    - Response times for performance monitoring
    """
    
    def __init__(self, app, exclude_paths: list = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/health", "/metrics"]
        self.exporter = get_prometheus_exporter()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and track metrics."""
        
        # Skip metrics collection for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Record start time
        start_time = time.time()
        
        # Process the request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Extract relevant information
            method = request.method
            endpoint = self._get_endpoint_template(request)
            status_code = response.status_code
            
            # Record metrics
            self.exporter.record_http_request(
                method=method,
                endpoint=endpoint, 
                status_code=status_code,
                duration=duration
            )
            
            # Add custom headers for observability  
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            
            return response
            
        except Exception as e:
            # Record error metrics
            duration = time.time() - start_time
            method = request.method
            endpoint = self._get_endpoint_template(request)
            
            self.exporter.record_http_request(
                method=method,
                endpoint=endpoint,
                status_code=500,
                duration=duration
            )
            
            logger.error("HTTP request failed", 
                        method=method,
                        endpoint=endpoint,
                        duration=duration,
                        error=str(e))
            
            raise
    
    def _get_endpoint_template(self, request: Request) -> str:
        """
        Extract the endpoint template for consistent metrics labeling.
        
        Converts dynamic paths like /api/v1/agents/123 to /api/v1/agents/{id}
        """
        path = request.url.path
        
        # Handle common API patterns
        if path.startswith("/api/v1/"):
            # Replace UUIDs and numeric IDs with placeholders
            import re
            # Replace UUIDs
            path = re.sub(r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '/{id}', path)
            # Replace numeric IDs
            path = re.sub(r'/\d+', '/{id}', path)
        
        # Handle WebSocket paths
        if path.startswith("/ws/"):
            return "/ws/*"
        
        # Handle static files
        if path.startswith("/static/"):
            return "/static/*"
        
        return path or "/"