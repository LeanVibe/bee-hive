"""
Security Orchestrator Integration for LeanVibe Agent Hive.

Integrates all security components with the existing orchestrator system,
providing seamless security enforcement and monitoring.

Features:
- OAuth 2.0/OIDC integration with orchestrator
- API security middleware integration
- Comprehensive audit logging integration
- RBAC enforcement in agent workflows
- Real-time security monitoring
- Compliance reporting automation
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import logging

import structlog
from fastapi import FastAPI, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from .oauth_provider_system import OAuthProviderSystem, OAuthProviderType
from .api_security_middleware import APISecurityMiddleware, SecurityConfig
from .comprehensive_audit_system import (
    ComprehensiveAuditSystem, AuditContext, AuditEventCategory, ComplianceFramework
)
from .authorization_engine import AuthorizationEngine
from .integrated_security_system import IntegratedSecuritySystem
from .redis import RedisClient
from .database import get_session
from ..schemas.security import OAuthProviderType as SchemaOAuthProviderType

logger = structlog.get_logger()


class SecurityOrchestrator:
    """
    Security Orchestrator for LeanVibe Agent Hive.
    
    Integrates all security components with the main orchestrator system,
    providing comprehensive security enforcement and monitoring.
    """
    
    def __init__(
        self,
        app: FastAPI,
        db_session: AsyncSession,
        redis_client: RedisClient,
        base_url: str = "http://localhost:8000"
    ):
        """
        Initialize Security Orchestrator.
        
        Args:
            app: FastAPI application instance
            db_session: Database session
            redis_client: Redis client
            base_url: Application base URL
        """
        self.app = app
        self.db = db_session
        self.redis = redis_client
        self.base_url = base_url
        
        # Security components
        self.oauth_system: Optional[OAuthProviderSystem] = None
        self.authorization_engine: Optional[AuthorizationEngine] = None
        self.audit_system: Optional[ComprehensiveAuditSystem] = None
        self.integrated_security: Optional[IntegratedSecuritySystem] = None
        self.api_security_middleware: Optional[APISecurityMiddleware] = None
        
        # Configuration
        self.security_config = SecurityConfig(
            enable_rate_limiting=True,
            default_rate_limit=100,
            enable_security_headers=True,
            enable_threat_detection=True,
            enable_sql_injection_detection=True,
            enable_xss_detection=True,
            max_request_size=10 * 1024 * 1024,  # 10MB
            log_security_events=True
        )
        
        # Metrics and monitoring
        self.metrics = {
            "security_events_processed": 0,
            "oauth_authentications": 0,
            "authorization_checks": 0,
            "audit_events_logged": 0,
            "security_violations_detected": 0,
            "compliance_reports_generated": 0
        }
        
        # Event handlers
        self._security_event_handlers: List[Callable] = []
        self._compliance_handlers: List[Callable] = []
    
    async def initialize(self) -> None:
        """Initialize all security components."""
        try:
            logger.info("Initializing Security Orchestrator")
            
            # Initialize OAuth provider system
            self.oauth_system = OAuthProviderSystem(
                db_session=self.db,
                redis_client=self.redis,
                base_url=self.base_url
            )
            
            # Initialize authorization engine
            self.authorization_engine = AuthorizationEngine(
                db_session=self.db,
                redis_client=self.redis
            )
            
            # Initialize comprehensive audit system
            self.audit_system = ComprehensiveAuditSystem(
                db_session=self.db,
                redis_client=self.redis,
                enabled_frameworks=[ComplianceFramework.SOC2, ComplianceFramework.ISO27001]
            )
            
            # Initialize API security middleware
            self.api_security_middleware = APISecurityMiddleware(
                app=self.app,
                redis_client=self.redis,
                config=self.security_config
            )
            
            # Add middleware to FastAPI app
            self.app.add_middleware(APISecurityMiddleware, 
                                   redis_client=self.redis, 
                                   config=self.security_config)
            
            # Configure default OAuth providers
            await self._configure_default_oauth_providers()
            
            # Set up security event monitoring
            await self._setup_security_monitoring()
            
            # Set up compliance monitoring
            await self._setup_compliance_monitoring()
            
            logger.info("Security Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Security Orchestrator: {e}")
            raise
    
    async def authenticate_request(
        self,
        request: Request,
        required_scopes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Authenticate incoming request using OAuth or JWT.
        
        Args:
            request: FastAPI request
            required_scopes: Required OAuth scopes
            
        Returns:
            Authentication result with user context
        """
        try:
            # Extract authorization header
            auth_header = request.headers.get("authorization")
            if not auth_header:
                return {"authenticated": False, "error": "No authorization header"}
            
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                
                # Try JWT validation first
                try:
                    from .security import verify_token
                    user_data = verify_token(token)
                    
                    # Log authentication event
                    await self._log_authentication_event(request, user_data, True)
                    
                    self.metrics["oauth_authentications"] += 1
                    
                    return {
                        "authenticated": True,
                        "user": user_data,
                        "method": "jwt"
                    }
                    
                except Exception:
                    # JWT validation failed, might be OAuth token
                    pass
            
            # Try OAuth token validation (if needed)
            # This would integrate with stored OAuth tokens
            
            await self._log_authentication_event(request, None, False)
            return {"authenticated": False, "error": "Invalid token"}
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            await self._log_authentication_event(request, None, False, str(e))
            return {"authenticated": False, "error": "Authentication error"}
    
    async def authorize_request(
        self,
        user_context: Dict[str, Any],
        resource: str,
        action: str,
        request_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Authorize request using RBAC system.
        
        Args:
            user_context: User context from authentication
            resource: Resource being accessed
            action: Action being performed
            request_context: Additional request context
            
        Returns:
            Authorization result
        """
        try:
            if not self.authorization_engine:
                return {"authorized": False, "error": "Authorization engine not initialized"}
            
            user_id = user_context.get("id") or user_context.get("sub")
            if not user_id:
                return {"authorized": False, "error": "Invalid user context"}
            
            # Perform authorization check
            auth_result = await self.authorization_engine.check_permission(
                agent_id=user_id,
                resource=resource,
                action=action,
                context=request_context or {}
            )
            
            # Log authorization event
            await self._log_authorization_event(user_id, resource, action, auth_result)
            
            self.metrics["authorization_checks"] += 1
            
            return {
                "authorized": auth_result.decision.value == "granted",
                "reason": auth_result.reason,
                "roles": auth_result.matched_roles,
                "permissions": auth_result.effective_permissions
            }
            
        except Exception as e:
            logger.error(f"Authorization failed: {e}")
            return {"authorized": False, "error": "Authorization error"}
    
    async def log_agent_action(
        self,
        agent_id: uuid.UUID,
        action: str,
        resource: Optional[str] = None,
        success: bool = True,
        request_data: Optional[Dict[str, Any]] = None,
        response_data: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[int] = None,
        client_ip: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> str:
        """
        Log agent action for audit trail.
        
        Args:
            agent_id: Agent identifier
            action: Action performed
            resource: Resource accessed
            success: Whether action was successful
            request_data: Request payload
            response_data: Response payload
            duration_ms: Action duration
            client_ip: Client IP address
            error_message: Error message if failed
            
        Returns:
            Audit event ID
        """
        try:
            if not self.audit_system:
                raise ValueError("Audit system not initialized")
            
            # Create audit context
            context = AuditContext(
                agent_id=agent_id,
                action=action,
                resource=resource,
                request_data=request_data,
                response_data=response_data,
                duration_ms=duration_ms,
                client_ip=client_ip,
                success=success,
                error_message=error_message,
                start_time=datetime.utcnow()
            )
            
            # Determine event category
            category = self._categorize_action(action)
            
            # Log audit event
            event_id = await self.audit_system.log_audit_event(context, category)
            
            self.metrics["audit_events_logged"] += 1
            
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to log agent action: {e}")
            raise
    
    async def detect_security_threat(
        self,
        agent_id: Optional[uuid.UUID],
        threat_type: str,
        description: str,
        severity: str = "medium",
        details: Optional[Dict[str, Any]] = None,
        source_ip: Optional[str] = None
    ) -> str:
        """
        Log security threat detection.
        
        Args:
            agent_id: Associated agent ID
            threat_type: Type of threat
            description: Threat description
            severity: Threat severity
            details: Additional details
            source_ip: Source IP address
            
        Returns:
            Security event ID
        """
        try:
            if not self.audit_system:
                raise ValueError("Audit system not initialized")
            
            from ..schemas.security import SecurityEventSeverityEnum
            severity_enum = SecurityEventSeverityEnum(severity)
            
            event_id = await self.audit_system.log_security_event(
                event_type=threat_type,
                severity=severity_enum,
                description=description,
                agent_id=agent_id,
                details=details,
                source_ip=source_ip
            )
            
            self.metrics["security_violations_detected"] += 1
            
            # Trigger security event handlers
            await self._trigger_security_event_handlers(threat_type, severity, details)
            
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to log security threat: {e}")
            raise
    
    async def generate_compliance_report(
        self,
        framework: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Generate compliance report.
        
        Args:
            framework: Compliance framework (soc2, iso27001, nist)
            start_date: Report start date
            end_date: Report end date
            
        Returns:
            Compliance report
        """
        try:
            if not self.audit_system:
                raise ValueError("Audit system not initialized")
            
            framework_enum = ComplianceFramework(framework)
            
            report = await self.audit_system.generate_compliance_report(
                framework=framework_enum,
                start_date=start_date,
                end_date=end_date
            )
            
            self.metrics["compliance_reports_generated"] += 1
            
            # Trigger compliance handlers
            await self._trigger_compliance_handlers(framework, report)
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate compliance report: {e}")
            raise
    
    async def configure_oauth_provider(
        self,
        provider_name: str,
        provider_type: str,
        client_id: str,
        client_secret: str,
        **kwargs
    ) -> bool:
        """
        Configure OAuth provider.
        
        Args:
            provider_name: Unique provider name
            provider_type: Provider type (google, github, microsoft, etc.)
            client_id: OAuth client ID
            client_secret: OAuth client secret
            **kwargs: Additional configuration
            
        Returns:
            True if configuration successful
        """
        try:
            if not self.oauth_system:
                raise ValueError("OAuth system not initialized")
            
            provider_type_enum = OAuthProviderType(provider_type)
            
            success = await self.oauth_system.configure_provider(
                provider_name=provider_name,
                provider_type=provider_type_enum,
                client_id=client_id,
                client_secret=client_secret,
                **kwargs
            )
            
            if success:
                logger.info(f"OAuth provider '{provider_name}' configured successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to configure OAuth provider: {e}")
            return False
    
    def add_security_event_handler(self, handler: Callable) -> None:
        """Add security event handler."""
        self._security_event_handlers.append(handler)
    
    def add_compliance_handler(self, handler: Callable) -> None:
        """Add compliance event handler."""
        self._compliance_handlers.append(handler)
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get comprehensive security metrics."""
        metrics = {
            "orchestrator_metrics": self.metrics.copy(),
            "oauth_metrics": self.oauth_system.get_metrics() if self.oauth_system else {},
            "authorization_metrics": self.authorization_engine.get_performance_metrics() if self.authorization_engine else {},
            "audit_metrics": self.audit_system.get_metrics() if self.audit_system else {},
            "api_security_metrics": self.api_security_middleware.get_metrics() if self.api_security_middleware else {}
        }
        
        return metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform security system health check."""
        health = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "components": {},
            "recommendations": []
        }
        
        try:
            # Check OAuth system
            if self.oauth_system:
                oauth_providers = self.oauth_system.get_provider_list()
                health["components"]["oauth"] = {
                    "status": "healthy",
                    "providers_configured": len(oauth_providers)
                }
            else:
                health["components"]["oauth"] = {"status": "not_initialized"}
            
            # Check authorization engine
            if self.authorization_engine:
                auth_metrics = await self.authorization_engine.get_performance_metrics()
                health["components"]["authorization"] = {
                    "status": "healthy",
                    "avg_evaluation_time_ms": auth_metrics["avg_evaluation_time_ms"]
                }
            else:
                health["components"]["authorization"] = {"status": "not_initialized"}
            
            # Check audit system
            if self.audit_system:
                audit_metrics = self.audit_system.get_metrics()
                health["components"]["audit"] = {
                    "status": "healthy",
                    "events_logged": audit_metrics["audit_system_metrics"]["events_logged"]
                }
            else:
                health["components"]["audit"] = {"status": "not_initialized"}
            
            # Check API security middleware
            if self.api_security_middleware:
                api_metrics = self.api_security_middleware.get_metrics()
                health["components"]["api_security"] = {
                    "status": "healthy",
                    "requests_processed": api_metrics["api_security_metrics"]["total_requests"]
                }
            else:
                health["components"]["api_security"] = {"status": "not_initialized"}
            
            # Generate recommendations
            if any(comp["status"] == "not_initialized" for comp in health["components"].values()):
                health["recommendations"].append("Some security components are not initialized")
                health["overall_status"] = "degraded"
            
            return health
            
        except Exception as e:
            logger.error(f"Security health check failed: {e}")
            health["overall_status"] = "error"
            health["error"] = str(e)
            return health
    
    # Private helper methods
    
    async def _configure_default_oauth_providers(self) -> None:
        """Configure default OAuth providers from environment."""
        import os
        
        # Google OAuth
        google_client_id = os.getenv("GOOGLE_OAUTH_CLIENT_ID")
        google_client_secret = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET")
        if google_client_id and google_client_secret:
            await self.configure_oauth_provider(
                provider_name="google",
                provider_type="google",
                client_id=google_client_id,
                client_secret=google_client_secret
            )
        
        # GitHub OAuth
        github_client_id = os.getenv("GITHUB_OAUTH_CLIENT_ID")
        github_client_secret = os.getenv("GITHUB_OAUTH_CLIENT_SECRET")
        if github_client_id and github_client_secret:
            await self.configure_oauth_provider(
                provider_name="github",
                provider_type="github",
                client_id=github_client_id,
                client_secret=github_client_secret
            )
        
        # Microsoft OAuth
        microsoft_client_id = os.getenv("MICROSOFT_OAUTH_CLIENT_ID")
        microsoft_client_secret = os.getenv("MICROSOFT_OAUTH_CLIENT_SECRET")
        microsoft_tenant_id = os.getenv("MICROSOFT_TENANT_ID")
        if microsoft_client_id and microsoft_client_secret:
            await self.configure_oauth_provider(
                provider_name="microsoft",
                provider_type="microsoft",
                client_id=microsoft_client_id,
                client_secret=microsoft_client_secret,
                tenant_id=microsoft_tenant_id
            )
    
    async def _setup_security_monitoring(self) -> None:
        """Set up real-time security monitoring."""
        try:
            # Add default security event handler
            async def default_security_handler(event_type: str, severity: str, details: Dict[str, Any]):
                logger.warning(f"Security event detected: {event_type}", 
                              severity=severity, details=details)
                
                # Store in Redis for external processing
                event_data = {
                    "type": event_type,
                    "severity": severity,
                    "details": details,
                    "timestamp": datetime.utcnow().isoformat()
                }
                await self.redis.lpush("security:events:real_time", json.dumps(event_data))
            
            self.add_security_event_handler(default_security_handler)
            
        except Exception as e:
            logger.error(f"Failed to set up security monitoring: {e}")
    
    async def _setup_compliance_monitoring(self) -> None:
        """Set up compliance monitoring and reporting."""
        try:
            # Add default compliance handler
            async def default_compliance_handler(framework: str, report: Dict[str, Any]):
                logger.info(f"Compliance report generated for {framework}", 
                           compliance_score=report.get("overall_compliance_score"))
                
                # Store compliance report
                await self.redis.set_with_expiry(
                    f"compliance:reports:{framework}:latest",
                    json.dumps(report),
                    ttl=86400 * 30  # 30 days
                )
            
            self.add_compliance_handler(default_compliance_handler)
            
        except Exception as e:
            logger.error(f"Failed to set up compliance monitoring: {e}")
    
    async def _log_authentication_event(
        self,
        request: Request,
        user_data: Optional[Dict[str, Any]],
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """Log authentication event for audit."""
        try:
            if not self.audit_system:
                return
            
            context = AuditContext(
                user_id=user_data.get("id") if user_data else None,
                action="authenticate",
                client_ip=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                success=success,
                error_message=error,
                authentication_method="oauth" if user_data else "unknown"
            )
            
            await self.audit_system.log_audit_event(context, AuditEventCategory.AUTHENTICATION)
            
        except Exception as e:
            logger.error(f"Failed to log authentication event: {e}")
    
    async def _log_authorization_event(
        self,
        user_id: str,
        resource: str,
        action: str,
        auth_result: Any
    ) -> None:
        """Log authorization event for audit."""
        try:
            if not self.audit_system:
                return
            
            context = AuditContext(
                user_id=user_id,
                action=f"authorize_{action}",
                resource=resource,
                success=auth_result.decision.value == "granted",
                authorization_result=auth_result.decision.value,
                permissions_checked=[f"{resource}:{action}"],
                roles_used=auth_result.matched_roles
            )
            
            await self.audit_system.log_audit_event(context, AuditEventCategory.AUTHORIZATION)
            
        except Exception as e:
            logger.error(f"Failed to log authorization event: {e}")
    
    def _categorize_action(self, action: str) -> AuditEventCategory:
        """Categorize action for audit logging."""
        action_lower = action.lower()
        
        if "auth" in action_lower:
            return AuditEventCategory.AUTHENTICATION
        elif "authorize" in action_lower or "permission" in action_lower:
            return AuditEventCategory.AUTHORIZATION
        elif "read" in action_lower or "get" in action_lower or "view" in action_lower:
            return AuditEventCategory.DATA_ACCESS
        elif "create" in action_lower or "update" in action_lower or "delete" in action_lower:
            return AuditEventCategory.DATA_MODIFICATION
        elif "admin" in action_lower or "config" in action_lower:
            return AuditEventCategory.SYSTEM_ADMINISTRATION
        elif "user" in action_lower or "role" in action_lower:
            return AuditEventCategory.USER_MANAGEMENT
        elif "privilege" in action_lower or "escalate" in action_lower:
            return AuditEventCategory.PRIVILEGE_ESCALATION
        else:
            return AuditEventCategory.DATA_ACCESS
    
    async def _trigger_security_event_handlers(
        self,
        event_type: str,
        severity: str,
        details: Optional[Dict[str, Any]]
    ) -> None:
        """Trigger security event handlers."""
        try:
            for handler in self._security_event_handlers:
                await handler(event_type, severity, details or {})
        except Exception as e:
            logger.error(f"Security event handler failed: {e}")
    
    async def _trigger_compliance_handlers(
        self,
        framework: str,
        report: Dict[str, Any]
    ) -> None:
        """Trigger compliance handlers."""
        try:
            for handler in self._compliance_handlers:
                await handler(framework, report)
        except Exception as e:
            logger.error(f"Compliance handler failed: {e}")
    
    async def cleanup(self) -> None:
        """Cleanup security orchestrator resources."""
        try:
            if self.oauth_system:
                await self.oauth_system.cleanup()
            
            if self.audit_system:
                await self.audit_system.cleanup()
            
            if self.api_security_middleware:
                await self.api_security_middleware.cleanup()
            
        except Exception as e:
            logger.error(f"Security orchestrator cleanup failed: {e}")


# Factory function
async def create_security_orchestrator(
    app: FastAPI,
    db_session: AsyncSession,
    redis_client: RedisClient,
    base_url: str = "http://localhost:8000"
) -> SecurityOrchestrator:
    """
    Create Security Orchestrator instance.
    
    Args:
        app: FastAPI application
        db_session: Database session
        redis_client: Redis client
        base_url: Application base URL
        
    Returns:
        SecurityOrchestrator instance
    """
    orchestrator = SecurityOrchestrator(app, db_session, redis_client, base_url)
    await orchestrator.initialize()
    return orchestrator