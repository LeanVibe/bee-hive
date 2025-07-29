"""
Enhanced Security Validation Middleware for LeanVibe Agent Hive 2.0.

Implements comprehensive input validation, sanitization, and security checks
for all incoming requests with performance optimization and threat detection.
"""

import re
import uuid
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import html
import urllib.parse
from datetime import datetime, timedelta

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

logger = structlog.get_logger()


class ThreatLevel(Enum):
    """Threat level enumeration."""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationResult(Enum):
    """Validation result enumeration."""
    VALID = "valid"
    SANITIZED = "sanitized"
    BLOCKED = "blocked"
    REJECTED = "rejected"


@dataclass
class SecurityValidationResult:
    """Result of security validation."""
    is_valid: bool
    validation_result: ValidationResult
    threat_level: ThreatLevel
    threats_detected: List[str]
    sanitized_data: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.0
    processing_time_ms: float = 0.0
    blocked_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "validation_result": self.validation_result.value,
            "threat_level": self.threat_level.value,
            "threats_detected": self.threats_detected,
            "confidence_score": self.confidence_score,
            "processing_time_ms": self.processing_time_ms,
            "blocked_reason": self.blocked_reason
        }


class SecurityValidationEngine:
    """
    Advanced security validation engine with multi-layer threat detection.
    
    Features:
    - SQL injection detection and prevention
    - XSS (Cross-Site Scripting) protection
    - Command injection prevention  
    - Path traversal detection
    - NoSQL injection protection
    - LDAP injection detection
    - Input sanitization and normalization
    - Content-based threat analysis
    - Performance-optimized pattern matching
    """
    
    def __init__(self):
        """Initialize security validation engine."""
        
        # SQL injection patterns (optimized regex)
        self.sql_injection_patterns = [
            re.compile(r"(\bUNION\b.*\bSELECT\b)", re.IGNORECASE),
            re.compile(r"(\bDROP\b.*\bTABLE\b)", re.IGNORECASE),
            re.compile(r"(\bINSERT\b.*\bINTO\b)", re.IGNORECASE),
            re.compile(r"(\bUPDATE\b.*\bSET\b)", re.IGNORECASE),
            re.compile(r"(\bDELETE\b.*\bFROM\b)", re.IGNORECASE),
            re.compile(r"(\bALTER\b.*\bTABLE\b)", re.IGNORECASE),
            re.compile(r"(\bCREATE\b.*\bTABLE\b)", re.IGNORECASE),
            re.compile(r"(\bEXEC\b.*\bSP_\w+)", re.IGNORECASE),
            re.compile(r"(\bSELECT\b.*\bFROM\b.*\bWHERE\b.*['\"].*['\"])", re.IGNORECASE),
            re.compile(r"(['\"];?\s*--)", re.IGNORECASE),
            re.compile(r"(['\"];\s*EXEC)", re.IGNORECASE),
            re.compile(r"(\bOR\b.*['\"].*['\"].*=.*['\"].*['\"])", re.IGNORECASE),
            re.compile(r"(\bAND\b.*['\"].*['\"].*=.*['\"].*['\"])", re.IGNORECASE),
            re.compile(r"(\\x[0-9A-Fa-f]{2})", re.IGNORECASE),  # Hex encoding
        ]
        
        # XSS patterns
        self.xss_patterns = [
            re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
            re.compile(r"<iframe[^>]*>.*?</iframe>", re.IGNORECASE | re.DOTALL),
            re.compile(r"<object[^>]*>.*?</object>", re.IGNORECASE | re.DOTALL),
            re.compile(r"<embed[^>]*>", re.IGNORECASE),
            re.compile(r"<link[^>]*>", re.IGNORECASE),
            re.compile(r"<meta[^>]*>", re.IGNORECASE),
            re.compile(r"on\w+\s*=", re.IGNORECASE),  # Event handlers
            re.compile(r"javascript:", re.IGNORECASE),
            re.compile(r"vbscript:", re.IGNORECASE),
            re.compile(r"data:text/html", re.IGNORECASE),
            re.compile(r"&lt;script", re.IGNORECASE),
            re.compile(r"&#x?[0-9a-f]+;", re.IGNORECASE),  # HTML entities
            re.compile(r"\\u[0-9a-f]{4}", re.IGNORECASE),  # Unicode escapes
        ]
        
        # Command injection patterns
        self.command_injection_patterns = [
            re.compile(r"[;&|`$(){}[\]\\]", re.IGNORECASE),
            re.compile(r"\$\(.*\)", re.IGNORECASE),  # Command substitution
            re.compile(r"`.*`", re.IGNORECASE),  # Backticks
            re.compile(r"\|\s*\w+", re.IGNORECASE),  # Pipes
            re.compile(r"&&\s*\w+", re.IGNORECASE),  # AND chains
            re.compile(r"\|\|\s*\w+", re.IGNORECASE),  # OR chains
            re.compile(r">\s*[/\w]+", re.IGNORECASE),  # Redirects
            re.compile(r"<\s*[/\w]+", re.IGNORECASE),  # Input redirects
            re.compile(r"\b(cat|ls|pwd|whoami|id|uname|ps|netstat|ifconfig|rm|mv|cp|chmod|chown|sudo|su)\b", re.IGNORECASE),
        ]
        
        # Path traversal patterns
        self.path_traversal_patterns = [
            re.compile(r"\.\./", re.IGNORECASE),
            re.compile(r"\.\.\\", re.IGNORECASE),
            re.compile(r"%2e%2e%2f", re.IGNORECASE),  # URL encoded ../
            re.compile(r"%2e%2e%5c", re.IGNORECASE),  # URL encoded ..\
            re.compile(r"\.%2e%2f", re.IGNORECASE),
            re.compile(r"%2e\.%2f", re.IGNORECASE),
            re.compile(r"%c0%ae%c0%ae%c0%af", re.IGNORECASE),  # UTF-8 overlong
            re.compile(r"%255c%255c", re.IGNORECASE),  # Double URL encoding
        ]
        
        # NoSQL injection patterns
        self.nosql_injection_patterns = [
            re.compile(r"\$where", re.IGNORECASE),
            re.compile(r"\$ne", re.IGNORECASE),
            re.compile(r"\$in", re.IGNORECASE),
            re.compile(r"\$nin", re.IGNORECASE),
            re.compile(r"\$gt", re.IGNORECASE),
            re.compile(r"\$lt", re.IGNORECASE),
            re.compile(r"\$regex", re.IGNORECASE),
            re.compile(r"\$or", re.IGNORECASE),
            re.compile(r"\$and", re.IGNORECASE),
            re.compile(r"\$javascript", re.IGNORECASE),
            re.compile(r"ObjectId\(", re.IGNORECASE),
        ]
        
        # LDAP injection patterns
        self.ldap_injection_patterns = [
            re.compile(r"[*()\\\x00]", re.IGNORECASE),
            re.compile(r"\\[0-9a-f]{2}", re.IGNORECASE),
        ]
        
        # Dangerous file extensions
        self.dangerous_extensions = {
            '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js', '.jar',
            '.ps1', '.sh', '.php', '.asp', '.aspx', '.jsp', '.py', '.rb', '.pl'
        }
        
        # Configuration
        self.config = {
            "max_input_length": 1000000,  # 1MB
            "max_nesting_depth": 10,
            "enable_content_sanitization": True,
            "enable_url_validation": True,
            "enable_file_validation": True,
            "blocked_content_types": ["application/x-executable", "application/octet-stream"],
            "allowed_domains": [],  # Empty means all allowed
            "rate_limit_per_minute": 1000,
            "enable_threat_scoring": True,
            "threat_threshold": 0.7
        }
        
        # Performance metrics
        self.metrics = {
            "total_validations": 0,
            "threats_detected": 0,
            "sanitized_inputs": 0,
            "blocked_requests": 0,
            "avg_processing_time_ms": 0.0,
            "threat_type_counts": {
                "sql_injection": 0,
                "xss": 0,
                "command_injection": 0,
                "path_traversal": 0,
                "nosql_injection": 0,
                "ldap_injection": 0,
                "file_upload": 0,
                "content_type": 0
            }
        }
    
    def validate_input(
        self, 
        data: Union[Dict[str, Any], str, List[Any]], 
        context: Optional[Dict[str, Any]] = None
    ) -> SecurityValidationResult:
        """
        Validate input data for security threats.
        
        Args:
            data: Input data to validate
            context: Additional context (request info, user data, etc.)
            
        Returns:
            SecurityValidationResult with validation outcome
        """
        start_time = time.time()
        
        try:
            self.metrics["total_validations"] += 1
            
            threats_detected = []
            threat_level = ThreatLevel.SAFE
            confidence_score = 0.0
            sanitized_data = None
            
            # Convert data to analyzable format
            if isinstance(data, dict):
                content_to_analyze = self._extract_content_from_dict(data)
            elif isinstance(data, list):
                content_to_analyze = self._extract_content_from_list(data)
            else:
                content_to_analyze = str(data)
            
            # Check input length
            if len(content_to_analyze) > self.config["max_input_length"]:
                return SecurityValidationResult(
                    is_valid=False,
                    validation_result=ValidationResult.BLOCKED,
                    threat_level=ThreatLevel.HIGH,
                    threats_detected=["input_too_large"],
                    confidence_score=1.0,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    blocked_reason=f"Input exceeds maximum length of {self.config['max_input_length']} characters"
                )
            
            # SQL injection detection
            sql_threats = self._detect_sql_injection(content_to_analyze)
            if sql_threats:
                threats_detected.extend(sql_threats)
                self.metrics["threat_type_counts"]["sql_injection"] += len(sql_threats)
                confidence_score = max(confidence_score, 0.9)
                threat_level = ThreatLevel.CRITICAL
            
            # XSS detection
            xss_threats = self._detect_xss(content_to_analyze)
            if xss_threats:
                threats_detected.extend(xss_threats)
                self.metrics["threat_type_counts"]["xss"] += len(xss_threats)
                confidence_score = max(confidence_score, 0.8)
                threat_level = max(threat_level, ThreatLevel.HIGH, key=lambda x: x.value)
            
            # Command injection detection
            cmd_threats = self._detect_command_injection(content_to_analyze)
            if cmd_threats:
                threats_detected.extend(cmd_threats)
                self.metrics["threat_type_counts"]["command_injection"] += len(cmd_threats)
                confidence_score = max(confidence_score, 0.9)
                threat_level = ThreatLevel.CRITICAL
            
            # Path traversal detection
            path_threats = self._detect_path_traversal(content_to_analyze)
            if path_threats:
                threats_detected.extend(path_threats)
                self.metrics["threat_type_counts"]["path_traversal"] += len(path_threats)
                confidence_score = max(confidence_score, 0.7)
                threat_level = max(threat_level, ThreatLevel.HIGH, key=lambda x: x.value)
            
            # NoSQL injection detection
            nosql_threats = self._detect_nosql_injection(content_to_analyze)
            if nosql_threats:
                threats_detected.extend(nosql_threats)
                self.metrics["threat_type_counts"]["nosql_injection"] += len(nosql_threats)
                confidence_score = max(confidence_score, 0.8)
                threat_level = max(threat_level, ThreatLevel.HIGH, key=lambda x: x.value)
            
            # LDAP injection detection
            ldap_threats = self._detect_ldap_injection(content_to_analyze)
            if ldap_threats:
                threats_detected.extend(ldap_threats)
                self.metrics["threat_type_counts"]["ldap_injection"] += len(ldap_threats)
                confidence_score = max(confidence_score, 0.7)
                threat_level = max(threat_level, ThreatLevel.MEDIUM, key=lambda x: x.value)
            
            # File upload validation
            if context and context.get("content_type"):
                file_threats = self._validate_file_upload(context)
                if file_threats:
                    threats_detected.extend(file_threats)
                    self.metrics["threat_type_counts"]["file_upload"] += len(file_threats)
                    confidence_score = max(confidence_score, 0.8)
                    threat_level = max(threat_level, ThreatLevel.HIGH, key=lambda x: x.value)
            
            # Determine final result
            processing_time = (time.time() - start_time) * 1000
            
            if threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
                self.metrics["blocked_requests"] += 1
                self.metrics["threats_detected"] += len(threats_detected)
                
                return SecurityValidationResult(
                    is_valid=False,
                    validation_result=ValidationResult.BLOCKED,
                    threat_level=threat_level,
                    threats_detected=threats_detected,
                    confidence_score=confidence_score,
                    processing_time_ms=processing_time,
                    blocked_reason=f"Security threats detected: {', '.join(threats_detected[:3])}"
                )
            
            elif threat_level == ThreatLevel.MEDIUM and self.config["enable_content_sanitization"]:
                # Attempt sanitization
                sanitized_data = self._sanitize_content(data)
                self.metrics["sanitized_inputs"] += 1
                
                return SecurityValidationResult(
                    is_valid=True,
                    validation_result=ValidationResult.SANITIZED,
                    threat_level=threat_level,
                    threats_detected=threats_detected,
                    sanitized_data=sanitized_data,
                    confidence_score=confidence_score,
                    processing_time_ms=processing_time
                )
            
            else:
                # Safe input
                return SecurityValidationResult(
                    is_valid=True,
                    validation_result=ValidationResult.VALID,
                    threat_level=threat_level,
                    threats_detected=threats_detected,
                    confidence_score=confidence_score,
                    processing_time_ms=processing_time
                )
            
        except Exception as e:
            logger.error(f"Security validation error: {e}")
            processing_time = (time.time() - start_time) * 1000
            
            return SecurityValidationResult(
                is_valid=False,
                validation_result=ValidationResult.REJECTED,
                threat_level=ThreatLevel.HIGH,
                threats_detected=["validation_error"],
                confidence_score=1.0,
                processing_time_ms=processing_time,
                blocked_reason=f"Validation processing error: {str(e)}"
            )
        
        finally:
            # Update average processing time
            current_avg = self.metrics["avg_processing_time_ms"]
            total_validations = self.metrics["total_validations"]
            new_time = (time.time() - start_time) * 1000
            self.metrics["avg_processing_time_ms"] = (
                (current_avg * (total_validations - 1) + new_time) / total_validations
            )
    
    def _extract_content_from_dict(self, data: Dict[str, Any], depth: int = 0) -> str:
        """Extract textual content from dictionary recursively."""
        if depth > self.config["max_nesting_depth"]:
            return ""
        
        content_parts = []
        for key, value in data.items():
            content_parts.append(str(key))
            
            if isinstance(value, dict):
                content_parts.append(self._extract_content_from_dict(value, depth + 1))
            elif isinstance(value, list):
                content_parts.append(self._extract_content_from_list(value, depth + 1))
            else:
                content_parts.append(str(value))
        
        return " ".join(content_parts)
    
    def _extract_content_from_list(self, data: List[Any], depth: int = 0) -> str:
        """Extract textual content from list recursively."""
        if depth > self.config["max_nesting_depth"]:
            return ""
        
        content_parts = []
        for item in data:
            if isinstance(item, dict):
                content_parts.append(self._extract_content_from_dict(item, depth + 1))
            elif isinstance(item, list):
                content_parts.append(self._extract_content_from_list(item, depth + 1))
            else:
                content_parts.append(str(item))
        
        return " ".join(content_parts)
    
    def _detect_sql_injection(self, content: str) -> List[str]:
        """Detect SQL injection patterns."""
        threats = []
        for pattern in self.sql_injection_patterns:
            if pattern.search(content):
                threats.append(f"sql_injection_{pattern.pattern[:20]}")
        return threats
    
    def _detect_xss(self, content: str) -> List[str]:
        """Detect XSS patterns."""
        threats = []
        for pattern in self.xss_patterns:
            if pattern.search(content):
                threats.append(f"xss_{pattern.pattern[:20]}")
        return threats
    
    def _detect_command_injection(self, content: str) -> List[str]:
        """Detect command injection patterns."""
        threats = []
        for pattern in self.command_injection_patterns:
            if pattern.search(content):
                threats.append(f"command_injection_{pattern.pattern[:20]}")
        return threats
    
    def _detect_path_traversal(self, content: str) -> List[str]:
        """Detect path traversal patterns."""
        threats = []
        for pattern in self.path_traversal_patterns:
            if pattern.search(content):
                threats.append(f"path_traversal_{pattern.pattern[:20]}")
        return threats
    
    def _detect_nosql_injection(self, content: str) -> List[str]:
        """Detect NoSQL injection patterns."""
        threats = []
        for pattern in self.nosql_injection_patterns:
            if pattern.search(content):
                threats.append(f"nosql_injection_{pattern.pattern[:20]}")
        return threats
    
    def _detect_ldap_injection(self, content: str) -> List[str]:
        """Detect LDAP injection patterns."""
        threats = []
        for pattern in self.ldap_injection_patterns:
            if pattern.search(content):
                threats.append(f"ldap_injection_{pattern.pattern[:20]}")
        return threats
    
    def _validate_file_upload(self, context: Dict[str, Any]) -> List[str]:
        """Validate file upload security."""
        threats = []
        
        content_type = context.get("content_type", "")
        filename = context.get("filename", "")
        
        # Check content type
        if content_type in self.config["blocked_content_types"]:
            threats.append(f"blocked_content_type_{content_type}")
        
        # Check file extension
        if filename:
            _, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            if f'.{ext.lower()}' in self.dangerous_extensions:
                threats.append(f"dangerous_file_extension_{ext}")
        
        return threats
    
    def _sanitize_content(self, data: Union[Dict[str, Any], str, List[Any]]) -> Any:
        """Sanitize content by removing/escaping dangerous patterns."""
        if isinstance(data, dict):
            return {k: self._sanitize_content(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_content(item) for item in data]
        elif isinstance(data, str):
            # HTML escape
            sanitized = html.escape(data)
            # URL decode
            sanitized = urllib.parse.unquote(sanitized)
            # Remove null bytes
            sanitized = sanitized.replace('\x00', '')
            return sanitized
        else:
            return data
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get validation metrics."""
        return {
            "performance_metrics": self.metrics.copy(),
            "config": self.config.copy(),
            "threat_detection_rate": (
                self.metrics["threats_detected"] / max(1, self.metrics["total_validations"])
            ),
            "sanitization_rate": (
                self.metrics["sanitized_inputs"] / max(1, self.metrics["total_validations"])
            ),
            "block_rate": (
                self.metrics["blocked_requests"] / max(1, self.metrics["total_validations"])
            )
        }


class SecurityValidationMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for security validation of all incoming requests.
    
    Validates request data, headers, and parameters before routing to endpoints.
    Provides comprehensive protection against common web vulnerabilities.
    """
    
    def __init__(self, app, validation_engine: Optional[SecurityValidationEngine] = None):
        """Initialize security validation middleware."""
        super().__init__(app)
        self.validation_engine = validation_engine or SecurityValidationEngine()
        
        # Paths to skip validation (health checks, static files, etc.)
        self.skip_validation_paths = {
            "/health",
            "/metrics", 
            "/docs",
            "/redoc",
            "/openapi.json",
            "/favicon.ico"
        }
        
        # Methods to validate
        self.validate_methods = {"POST", "PUT", "PATCH", "DELETE"}
        
        # Rate limiting storage (in production, use Redis)
        self.rate_limit_storage = {}
    
    async def dispatch(self, request: Request, call_next):
        """Process request through security validation."""
        start_time = time.time()
        
        try:
            # Skip validation for certain paths
            if any(request.url.path.startswith(path) for path in self.skip_validation_paths):
                return await call_next(request)
            
            # Rate limiting check
            client_ip = self._get_client_ip(request)
            if not self._check_rate_limit(client_ip):
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "rate_limit_exceeded",
                        "message": "Too many requests from this IP address",
                        "retry_after": 60
                    }
                )
            
            # Only validate certain HTTP methods
            if request.method in self.validate_methods:
                validation_result = await self._validate_request(request)
                
                if not validation_result.is_valid:
                    # Log security event
                    logger.warning(
                        "Security validation failed",
                        client_ip=client_ip,
                        path=request.url.path,
                        method=request.method,
                        threats=validation_result.threats_detected,
                        threat_level=validation_result.threat_level.value,
                        blocked_reason=validation_result.blocked_reason
                    )
                    
                    # Return security error
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={
                            "error": "security_validation_failed",
                            "message": "Request blocked due to security concerns",
                            "threat_level": validation_result.threat_level.value,
                            "correlation_id": str(uuid.uuid4())
                        }
                    )
                
                # If sanitized, modify request data
                if validation_result.validation_result == ValidationResult.SANITIZED:
                    # In production, would modify request body with sanitized data
                    logger.info(
                        "Request sanitized",
                        client_ip=client_ip,
                        path=request.url.path,
                        threats=validation_result.threats_detected
                    )
            
            # Process request
            response = await call_next(request)
            
            # Add security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
            
            # Add CSP header
            csp = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self'; "
                "connect-src 'self' wss: ws:; "
                "frame-ancestors 'none'"
            )
            response.headers["Content-Security-Policy"] = csp
            
            return response
            
        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "security_middleware_error",
                    "message": "Internal security validation error"
                }
            )
    
    async def _validate_request(self, request: Request) -> SecurityValidationResult:
        """Validate incoming request."""
        try:
            # Prepare request context
            context = {
                "method": request.method,
                "path": request.url.path,
                "client_ip": self._get_client_ip(request),
                "user_agent": request.headers.get("user-agent", ""),
                "content_type": request.headers.get("content-type", ""),
                "content_length": request.headers.get("content-length", "0"),
                "referer": request.headers.get("referer", ""),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Validate headers
            header_validation = self.validation_engine.validate_input(
                dict(request.headers), context
            )
            
            if not header_validation.is_valid:
                return header_validation
            
            # Validate query parameters
            if request.query_params:
                query_validation = self.validation_engine.validate_input(
                    dict(request.query_params), context
                )
                
                if not query_validation.is_valid:
                    return query_validation
            
            # Validate request body for applicable methods
            if request.method in {"POST", "PUT", "PATCH"}:
                try:
                    body = await request.body()
                    if body:
                        # Try to parse as JSON
                        try:
                            json_data = json.loads(body.decode('utf-8'))
                            body_validation = self.validation_engine.validate_input(
                                json_data, context
                            )
                        except json.JSONDecodeError:
                            # Validate raw body
                            body_validation = self.validation_engine.validate_input(
                                body.decode('utf-8', errors='ignore'), context
                            )
                        
                        if not body_validation.is_valid:
                            return body_validation
                            
                except Exception as e:
                    logger.debug(f"Body validation error: {e}")
                    # If we can't read body, allow request to proceed
                    pass
            
            # If all validations pass, return success
            return SecurityValidationResult(
                is_valid=True,
                validation_result=ValidationResult.VALID,
                threat_level=ThreatLevel.SAFE,
                threats_detected=[],
                confidence_score=1.0,
                processing_time_ms=0.0
            )
            
        except Exception as e:
            logger.error(f"Request validation error: {e}")
            return SecurityValidationResult(
                is_valid=False,
                validation_result=ValidationResult.REJECTED,
                threat_level=ThreatLevel.HIGH,
                threats_detected=["validation_error"],
                confidence_score=1.0,
                processing_time_ms=0.0,
                blocked_reason=f"Request validation error: {str(e)}"
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        # Check for forwarded headers (load balancer/proxy)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fall back to direct connection
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _check_rate_limit(self, client_ip: str) -> bool:
        """Simple rate limiting check."""
        now = datetime.utcnow()
        minute_key = f"{client_ip}:{now.strftime('%Y%m%d%H%M')}"
        
        # Clean old entries (simplified - use Redis with TTL in production)
        cutoff = now - timedelta(minutes=2)
        keys_to_remove = [
            k for k in self.rate_limit_storage.keys()
            if datetime.strptime(k.split(':')[1], '%Y%m%d%H%M') < cutoff
        ]
        for k in keys_to_remove:
            del self.rate_limit_storage[k]
        
        # Check current minute
        current_count = self.rate_limit_storage.get(minute_key, 0)
        if current_count >= self.validation_engine.config["rate_limit_per_minute"]:
            return False
        
        # Increment counter
        self.rate_limit_storage[minute_key] = current_count + 1
        return True


# Factory function
def create_security_validation_middleware() -> SecurityValidationMiddleware:
    """Create security validation middleware instance."""
    return SecurityValidationMiddleware(None)  # App will be set by FastAPI