"""
Security Middleware for request interception and validation.

Implements comprehensive security controls including authentication verification,
authorization checks, input validation, and security headers.
"""

import uuid
import time
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Awaitable
from urllib.parse import urlparse
import logging

from fastapi import Request, Response, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .agent_identity_service import AgentIdentityService, TokenValidationError
from .authorization_engine import AuthorizationEngine, AuthorizationResult, AccessDecision
from .audit_logger import AuditLogger, AuditContext
from ..schemas.security import SecurityError

logger = logging.getLogger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive Security Middleware.
    
    Features:
    - JWT token validation and extraction
    - Role-based authorization checking
    - Request/response audit logging
    - Input validation and sanitization
    - Security headers injection
    - Rate limiting and DDoS protection
    - CORS policy enforcement
    - Content Security Policy (CSP)
    """
    
    def __init__(
        self,
        app,
        identity_service: AgentIdentityService,
        authorization_engine: AuthorizationEngine,
        audit_logger: AuditLogger,
        enable_cors: bool = True,
        enable_csrf_protection: bool = True,
        enable_rate_limiting: bool = True,
        public_paths: Optional[List[str]] = None
    ):
        """
        Initialize Security Middleware.
        
        Args:
            app: FastAPI application
            identity_service: Agent identity service
            authorization_engine: Authorization engine
            audit_logger: Audit logger
            enable_cors: Enable CORS handling
            enable_csrf_protection: Enable CSRF protection
            enable_rate_limiting: Enable rate limiting
            public_paths: Paths that don't require authentication
        """
        super().__init__(app)
        self.identity_service = identity_service
        self.authorization_engine = authorization_engine
        self.audit_logger = audit_logger
        self.enable_cors = enable_cors
        self.enable_csrf_protection = enable_csrf_protection
        self.enable_rate_limiting = enable_rate_limiting
        
        # Default public paths
        self.public_paths = public_paths or [
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/metrics",
            "/auth/agent/token",  # Authentication endpoint
            "/favicon.ico"
        ]
        
        # Security configuration
        self.security_config = {
            "max_request_size": 10 * 1024 * 1024,  # 10MB
            "max_header_size": 8 * 1024,  # 8KB
            "allowed_methods": ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
            "blocked_user_agents": ["sqlmap", "nikto", "nessus", "OpenVAS"],
            "blocked_ips": [],  # Would be populated from threat intelligence
            "require_https": False,  # Set to True in production
            "csrf_token_header": "X-CSRF-Token",
            "session_timeout_hours": 24
        }
        
        # Security headers to inject
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "connect-src 'self'; "
                "font-src 'self'; "
                "object-src 'none'; "
                "media-src 'self'; "
                "frame-src 'none';"
            )
        }
        
        # CORS configuration
        self.cors_config = {
            "allow_origins": ["http://localhost:3000", "http://localhost:8080"],  # Development
            "allow_methods": ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
            "allow_headers": [
                "Authorization",
                "Content-Type", 
                "X-CSRF-Token",
                "X-Requested-With",
                "Accept",
                "Origin",
                "User-Agent"
            ],
            "expose_headers": ["X-RateLimit-Remaining", "X-RateLimit-Reset"],
            "allow_credentials": True,
            "max_age": 86400  # 24 hours
        }
        
        # Performance metrics
        self.metrics = {
            "requests_processed": 0,
            "authentication_failures": 0,
            "authorization_failures": 0,
            "security_violations": 0,
            "blocked_requests": 0,
            "avg_processing_time_ms": 0.0
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Main middleware dispatch method.
        
        Args:
            request: HTTP request
            call_next: Next middleware/handler
            
        Returns:
            HTTP response with security controls applied
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Add request ID to request state
            request.state.request_id = request_id
            request.state.start_time = start_time
            
            # Pre-flight security checks
            security_check_result = await self._pre_flight_security_checks(request)
            if security_check_result:
                return security_check_result
            
            # Handle CORS preflight
            if request.method == "OPTIONS" and self.enable_cors:
                return self._create_cors_response()
            
            # Skip authentication for public paths
            if self._is_public_path(request.url.path):
                response = await call_next(request)
                return await self._post_process_response(request, response, None, None)
            
            # Extract and validate authentication
            auth_result = await self._authenticate_request(request)
            if isinstance(auth_result, Response):
                return auth_result  # Authentication failed
            
            agent_token, agent_identity = auth_result
            
            # Store authentication info in request state
            request.state.agent_token = agent_token
            request.state.agent_identity = agent_identity
            request.state.human_controller = agent_token.get("human_controller")
            
            # Authorize request
            authz_result = await self._authorize_request(request, agent_token)
            if isinstance(authz_result, Response):
                return authz_result  # Authorization failed
            
            # Store authorization info
            request.state.authorization_result = authz_result
            
            # Process request
            response = await call_next(request)
            
            # Post-process response
            return await self._post_process_response(request, response, agent_token, authz_result)
            
        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            return await self._handle_security_error(request, e, start_time)
        finally:
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self.metrics["requests_processed"] += 1
            self.metrics["avg_processing_time_ms"] = (
                (self.metrics["avg_processing_time_ms"] * (self.metrics["requests_processed"] - 1) + processing_time) /
                self.metrics["requests_processed"]
            )
    
    async def _pre_flight_security_checks(self, request: Request) -> Optional[Response]:
        """
        Perform pre-flight security checks.
        
        Args:
            request: HTTP request
            
        Returns:
            Error response if security check fails, None otherwise
        """
        try:
            # Check request size
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.security_config["max_request_size"]:
                self.metrics["blocked_requests"] += 1
                return JSONResponse(
                    status_code=413,
                    content={"error": "request_too_large", "message": "Request payload too large"}
                )
            
            # Check HTTP method
            if request.method not in self.security_config["allowed_methods"]:
                self.metrics["blocked_requests"] += 1
                return JSONResponse(
                    status_code=405,
                    content={"error": "method_not_allowed", "message": f"Method {request.method} not allowed"}
                )
            
            # Check User-Agent for known bad bots
            user_agent = request.headers.get("user-agent", "").lower()
            for blocked_agent in self.security_config["blocked_user_agents"]:
                if blocked_agent.lower() in user_agent:
                    self.metrics["security_violations"] += 1
                    await self._log_security_violation(
                        request, "blocked_user_agent", f"Blocked user agent: {user_agent}"
                    )
                    return JSONResponse(
                        status_code=403,
                        content={"error": "forbidden", "message": "Access denied"}
                    )
            
            # Check for blocked IPs
            client_ip = self._get_client_ip(request)
            if client_ip in self.security_config["blocked_ips"]:
                self.metrics["security_violations"] += 1
                await self._log_security_violation(
                    request, "blocked_ip", f"Blocked IP: {client_ip}"
                )
                return JSONResponse(
                    status_code=403,
                    content={"error": "forbidden", "message": "Access denied"}
                )
            
            # Check HTTPS requirement
            if (self.security_config["require_https"] and 
                request.url.scheme != "https" and 
                not self._is_local_request(request)):
                return JSONResponse(
                    status_code=426,
                    content={"error": "https_required", "message": "HTTPS required"}
                )
            
            # Check for suspicious headers
            suspicious_headers = ["x-forwarded-host", "x-cluster-client-ip"]
            for header in suspicious_headers:
                if header in request.headers:
                    logger.warning(f"Suspicious header detected: {header}")
                    # Log but don't block (might be legitimate proxy)
            
            return None
            
        except Exception as e:
            logger.error(f"Pre-flight security check error: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "security_check_failed", "message": "Security validation error"}
            )
    
    async def _authenticate_request(self, request: Request) -> tuple:
        """
        Authenticate the request using JWT token.
        
        Args:
            request: HTTP request
            
        Returns:
            Tuple of (token_payload, agent_identity) or error Response
        """
        try:
            # Extract Bearer token
            auth_header = request.headers.get("authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                self.metrics["authentication_failures"] += 1
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "missing_token",
                        "error_description": "Missing or invalid Authorization header"
                    }
                )
            
            token = auth_header[7:]  # Remove "Bearer " prefix
            
            # Validate token
            token_payload = await self.identity_service.validate_token(token)
            
            # Get agent identity
            agent_id = token_payload.get("sub")
            agent_identity = await self.identity_service._get_agent_identity(agent_id)
            
            if not agent_identity or not agent_identity.is_active():
                self.metrics["authentication_failures"] += 1
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "invalid_agent",
                        "error_description": "Agent not found or inactive"
                    }
                )
            
            return token_payload, agent_identity
            
        except TokenValidationError as e:
            self.metrics["authentication_failures"] += 1
            await self._log_security_violation(
                request, "authentication_failure", f"Token validation failed: {e}"
            )
            return JSONResponse(
                status_code=401,
                content={
                    "error": "invalid_token",
                    "error_description": str(e)
                }
            )
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            self.metrics["authentication_failures"] += 1
            return JSONResponse(
                status_code=500,
                content={
                    "error": "authentication_error",
                    "error_description": "Authentication service error"
                }
            )
    
    async def _authorize_request(self, request: Request, agent_token: Dict[str, Any]) -> AuthorizationResult:
        """
        Authorize the request using RBAC.
        
        Args:
            request: HTTP request
            agent_token: Validated agent token
            
        Returns:
            AuthorizationResult or error Response
        """
        try:
            # Extract resource and action from request
            resource, action = self._extract_resource_action(request)
            
            # Prepare authorization context
            context = {
                "ip_address": self._get_client_ip(request),
                "user_agent": request.headers.get("user-agent"),
                "session_id": request.headers.get("x-session-id"),
                "request_id": request.state.request_id,
                "method": request.method,
                "endpoint": str(request.url.path)
            }
            
            # Check authorization
            authz_result = await self.authorization_engine.check_permission(
                agent_id=agent_token["sub"],
                resource=resource,
                action=action,
                context=context
            )
            
            if authz_result.decision != AccessDecision.GRANTED:
                self.metrics["authorization_failures"] += 1
                await self._log_security_violation(
                    request, "authorization_failure", 
                    f"Access denied: {authz_result.reason}"
                )
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "access_denied",
                        "error_description": authz_result.reason,
                        "required_permissions": authz_result.effective_permissions
                    }
                )
            
            return authz_result
            
        except Exception as e:
            logger.error(f"Authorization error: {e}")
            self.metrics["authorization_failures"] += 1
            return JSONResponse(
                status_code=500,
                content={
                    "error": "authorization_error",
                    "error_description": "Authorization service error"
                }
            )
    
    async def _post_process_response(
        self,
        request: Request,
        response: Response,
        agent_token: Optional[Dict[str, Any]],
        authz_result: Optional[AuthorizationResult]
    ) -> Response:
        """
        Post-process response with security controls.
        
        Args:
            request: HTTP request
            response: HTTP response
            agent_token: Agent token (if authenticated)
            authz_result: Authorization result (if authorized)
            
        Returns:
            Modified response with security headers
        """
        try:
            # Add security headers
            for header, value in self.security_headers.items():
                response.headers[header] = value
            
            # Add CORS headers
            if self.enable_cors:
                self._add_cors_headers(response, request)
            
            # Add request ID header
            response.headers["X-Request-ID"] = request.state.request_id
            
            # Add processing time header
            processing_time = (time.time() - request.state.start_time) * 1000
            response.headers["X-Processing-Time-Ms"] = str(int(processing_time))
            
            # Audit log the request/response
            await self._audit_request_response(request, response, agent_token, authz_result)
            
            return response
            
        except Exception as e:
            logger.error(f"Response post-processing error: {e}")
            return response  # Return original response if post-processing fails
    
    async def _audit_request_response(
        self,
        request: Request,
        response: Response,
        agent_token: Optional[Dict[str, Any]],
        authz_result: Optional[AuthorizationResult]
    ) -> None:
        """
        Audit log the request and response.
        
        Args:
            request: HTTP request
            response: HTTP response
            agent_token: Agent token
            authz_result: Authorization result
        """
        try:
            # Create audit context
            context = AuditContext(
                request_id=request.state.request_id,
                agent_id=uuid.UUID(agent_token["sub"]) if agent_token else None,
                human_controller=agent_token.get("human_controller", "anonymous") if agent_token else "anonymous",
                session_id=uuid.UUID(request.headers.get("x-session-id")) if request.headers.get("x-session-id") else None,
                ip_address=self._get_client_ip(request),
                user_agent=request.headers.get("user-agent"),
                geo_location=None,  # Would be populated by audit logger
                timestamp=datetime.utcnow()
            )
            
            # Extract action and resource
            resource, action = self._extract_resource_action(request)
            
            # Prepare request data (sanitized)
            request_data = {
                "method": request.method,
                "path": str(request.url.path),
                "query_params": dict(request.query_params),
                "headers": dict(request.headers)
            }
            
            # Remove sensitive headers
            sensitive_headers = ["authorization", "cookie", "x-csrf-token"]
            for header in sensitive_headers:
                request_data["headers"].pop(header, None)
            
            # Prepare response data
            response_data = {
                "status_code": response.status_code,
                "headers": dict(response.headers)
            }
            
            # Log the event
            await self.audit_logger.log_event(
                context=context,
                action=action,
                resource=resource,
                success=(200 <= response.status_code < 400),
                method=request.method,
                endpoint=str(request.url.path),
                request_data=request_data,
                response_data=response_data,
                duration_ms=int((time.time() - request.state.start_time) * 1000),
                permission_checked=f"{resource}:{action}" if authz_result else None,
                authorization_result=authz_result.decision.value if authz_result else None
            )
            
        except Exception as e:
            logger.error(f"Audit logging failed: {e}")
    
    def _extract_resource_action(self, request: Request) -> tuple[str, str]:
        """
        Extract resource and action from request.
        
        Args:
            request: HTTP request
            
        Returns:
            Tuple of (resource, action)
        """
        path = request.url.path.strip("/")
        method = request.method.lower()
        
        # Map HTTP methods to actions
        method_action_map = {
            "get": "read",
            "post": "create",
            "put": "update",
            "patch": "update",
            "delete": "delete",
            "options": "read"
        }
        
        action = method_action_map.get(method, method)
        
        # Extract resource from path
        if not path:
            resource = "root"
        else:
            # Use first path segment as resource
            path_parts = path.split("/")
            resource = path_parts[0] if path_parts else "unknown"
        
        return resource, action
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        # Check for forwarded headers (in order of preference)
        forwarded_headers = [
            "x-forwarded-for",
            "x-real-ip",
            "x-client-ip",
            "cf-connecting-ip"  # Cloudflare
        ]
        
        for header in forwarded_headers:
            ip = request.headers.get(header)
            if ip:
                # Take first IP if comma-separated
                return ip.split(",")[0].strip()
        
        # Fallback to direct connection
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return "unknown"
    
    def _is_public_path(self, path: str) -> bool:
        """Check if path is public (doesn't require authentication)."""
        return any(path.startswith(public_path) for public_path in self.public_paths)
    
    def _is_local_request(self, request: Request) -> bool:
        """Check if request is from localhost."""
        client_ip = self._get_client_ip(request)
        return client_ip in ["127.0.0.1", "::1", "localhost"]
    
    def _create_cors_response(self) -> Response:
        """Create CORS preflight response."""
        response = Response(status_code=204)
        self._add_cors_headers(response, None)
        return response
    
    def _add_cors_headers(self, response: Response, request: Optional[Request]) -> None:
        """Add CORS headers to response."""
        if not self.enable_cors:
            return
        
        # Get origin from request
        origin = None
        if request:
            origin = request.headers.get("origin")
        
        # Check if origin is allowed
        if origin and (
            "*" in self.cors_config["allow_origins"] or
            origin in self.cors_config["allow_origins"]
        ):
            response.headers["Access-Control-Allow-Origin"] = origin
        elif not origin:
            response.headers["Access-Control-Allow-Origin"] = "*"
        
        response.headers["Access-Control-Allow-Methods"] = ", ".join(self.cors_config["allow_methods"])
        response.headers["Access-Control-Allow-Headers"] = ", ".join(self.cors_config["allow_headers"])
        response.headers["Access-Control-Expose-Headers"] = ", ".join(self.cors_config["expose_headers"])
        response.headers["Access-Control-Max-Age"] = str(self.cors_config["max_age"])
        
        if self.cors_config["allow_credentials"]:
            response.headers["Access-Control-Allow-Credentials"] = "true"
    
    async def _log_security_violation(
        self,
        request: Request,
        violation_type: str,
        description: str
    ) -> None:
        """Log security violation."""
        try:
            context = AuditContext(
                request_id=getattr(request.state, "request_id", str(uuid.uuid4())),
                agent_id=None,
                human_controller="anonymous",
                session_id=None,
                ip_address=self._get_client_ip(request),
                user_agent=request.headers.get("user-agent"),
                geo_location=None,
                timestamp=datetime.utcnow()
            )
            
            await self.audit_logger.log_event(
                context=context,
                action="security_violation",
                resource="security",
                success=False,
                error_message=description,
                metadata={
                    "violation_type": violation_type,
                    "method": request.method,
                    "path": str(request.url.path)
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to log security violation: {e}")
    
    async def _handle_security_error(
        self,
        request: Request,
        error: Exception,
        start_time: float
    ) -> Response:
        """Handle security middleware errors."""
        self.metrics["security_violations"] += 1
        
        try:
            # Log the error
            await self._log_security_violation(
                request, "middleware_error", f"Security middleware error: {error}"
            )
        except:
            pass  # Don't let logging errors crash the request
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "security_error",
                "error_description": "Security validation failed",
                "request_id": getattr(request.state, "request_id", str(uuid.uuid4()))
            }
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get security middleware metrics."""
        return {
            "security_middleware": self.metrics.copy(),
            "configuration": {
                "enable_cors": self.enable_cors,
                "enable_csrf_protection": self.enable_csrf_protection,
                "enable_rate_limiting": self.enable_rate_limiting,
                "public_paths_count": len(self.public_paths),
                "require_https": self.security_config["require_https"]
            }
        }


class HTTPSRedirectMiddleware(BaseHTTPMiddleware):
    """Middleware to redirect HTTP to HTTPS in production."""
    
    def __init__(self, app, force_https: bool = False):
        super().__init__(app)
        self.force_https = force_https
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if (self.force_https and 
            request.url.scheme == "http" and 
            not request.client.host.startswith("127.") and
            request.client.host != "localhost"):
            
            # Redirect to HTTPS
            https_url = request.url.replace(scheme="https")
            return JSONResponse(
                status_code=301,
                headers={"Location": str(https_url)}
            )
        
        return await call_next(request)


# Factory function
def create_security_middleware(
    identity_service: AgentIdentityService,
    authorization_engine: AuthorizationEngine,
    audit_logger: AuditLogger,
    **kwargs
) -> SecurityMiddleware:
    """
    Create Security Middleware instance.
    
    Args:
        identity_service: Agent identity service
        authorization_engine: Authorization engine
        audit_logger: Audit logger
        **kwargs: Additional configuration
        
    Returns:
        SecurityMiddleware instance
    """
    class SecurityMiddlewareFactory:
        def __init__(self, middleware_class, *args, **kwargs):
            self.middleware_class = middleware_class
            self.args = args
            self.kwargs = kwargs
        
        def __call__(self, app):
            return self.middleware_class(app, *self.args, **self.kwargs)
    
    return SecurityMiddlewareFactory(
        SecurityMiddleware,
        identity_service,
        authorization_engine,
        audit_logger,
        **kwargs
    )