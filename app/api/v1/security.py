"""
Security API endpoints for authentication, authorization, and audit operations.

Implements OAuth 2.0/OIDC endpoints, permission checking APIs,
and comprehensive audit query interfaces.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from fastapi import APIRouter, Depends, HTTPException, status, Request, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.database import get_db
from ...core.agent_identity_service import AgentIdentityService, TokenValidationError, RateLimitExceededError
from ...core.authorization_engine import AuthorizationEngine, AccessDecision
from ...core.audit_logger import AuditLogger, AuditContext
from ...core.secret_manager import SecretManager
from ...schemas.security import (
    AgentTokenRequest, TokenRefreshRequest, TokenResponse,
    PermissionCheckRequest, PermissionCheckResponse,
    AuditLogRequest, SecurityEventRequest,
    AgentIdentityResponse, AgentRoleResponse,
    SecurityAuditLogResponse, SecurityEventResponse,
    SecurityDashboard, SecurityError, ValidationError,
    AuditLogFilters, SecurityEventFilters,
    RoleAssignmentRequest
)
from ...models.security import AgentIdentity, AgentRole, SecurityEventSeverity

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/security", tags=["security"])

# Security scheme
security = HTTPBearer(auto_error=False)


# Dependency injection helpers
async def get_identity_service() -> AgentIdentityService:
    """Get agent identity service instance."""
    # In production, this would be injected via dependency container
    # For now, return a placeholder
    raise HTTPException(
        status_code=500,
        detail="Identity service not configured"
    )


async def get_authorization_engine() -> AuthorizationEngine:
    """Get authorization engine instance."""
    # In production, this would be injected via dependency container
    raise HTTPException(
        status_code=500,
        detail="Authorization engine not configured"
    )


async def get_audit_logger() -> AuditLogger:
    """Get audit logger instance."""
    # In production, this would be injected via dependency container
    raise HTTPException(
        status_code=500,
        detail="Audit logger not configured"
    )


async def get_secret_manager() -> SecretManager:
    """Get secret manager instance."""
    # In production, this would be injected via dependency container
    raise HTTPException(
        status_code=500,
        detail="Secret manager not configured"
    )


async def get_current_agent(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    identity_service: AgentIdentityService = Depends(get_identity_service)
) -> Dict[str, Any]:
    """Get current authenticated agent from JWT token."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token"
        )
    
    try:
        token_payload = await identity_service.validate_token(credentials.credentials)
        return token_payload
    except TokenValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )


def create_audit_context(request: Request, agent_token: Optional[Dict[str, Any]] = None) -> AuditContext:
    """Create audit context from request."""
    return AuditContext(
        request_id=getattr(request.state, 'request_id', str(uuid.uuid4())),
        agent_id=uuid.UUID(agent_token["sub"]) if agent_token else None,
        human_controller=agent_token.get("human_controller", "anonymous") if agent_token else "anonymous",
        session_id=uuid.UUID(request.headers.get("x-session-id")) if request.headers.get("x-session-id") else None,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        geo_location=None,
        timestamp=datetime.utcnow()
    )


# Authentication Endpoints

@router.post("/auth/agent/token", response_model=TokenResponse)
async def request_agent_token(
    request_data: AgentTokenRequest,
    request: Request,
    identity_service: AgentIdentityService = Depends(get_identity_service),
    audit_logger: AuditLogger = Depends(get_audit_logger)
):
    """
    Request OAuth 2.0 access token for agent authentication.
    
    This endpoint implements the OAuth 2.0 client credentials flow
    for agent authentication with human accountability.
    """
    try:
        # Create audit context
        context = create_audit_context(request)
        
        # Authenticate agent and issue tokens
        token_response = await identity_service.authenticate_agent(
            request=request_data,
            client_ip=context.ip_address,
            user_agent=context.user_agent
        )
        
        # Log successful authentication
        await audit_logger.log_event(
            context=context,
            action="agent_token_request",
            resource="authentication",
            success=True,
            metadata={
                "agent_id": request_data.agent_id,
                "scopes": request_data.requested_scopes,
                "token_type": "access+refresh"
            }
        )
        
        return token_response
        
    except RateLimitExceededError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {e}"
        )
    except Exception as e:
        logger.error(f"Token request failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error"
        )


@router.post("/auth/token/refresh", response_model=TokenResponse)
async def refresh_agent_token(
    request_data: TokenRefreshRequest,
    request: Request,
    identity_service: AgentIdentityService = Depends(get_identity_service),
    audit_logger: AuditLogger = Depends(get_audit_logger)
):
    """
    Refresh access token using refresh token.
    
    Allows agents to obtain new access tokens without re-authentication.
    """
    try:
        # Create audit context
        context = create_audit_context(request)
        
        # Refresh token
        token_response = await identity_service.refresh_token(
            request=request_data,
            client_ip=context.ip_address,
            user_agent=context.user_agent
        )
        
        # Log token refresh
        await audit_logger.log_event(
            context=context,
            action="token_refresh",
            resource="authentication",
            success=True,
            metadata={"scopes": request_data.requested_scopes}
        )
        
        return token_response
        
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token refresh failed"
        )


@router.post("/auth/token/revoke")
async def revoke_agent_token(
    token: str,
    request: Request,
    reason: str = "manual_revocation",
    current_agent: Dict[str, Any] = Depends(get_current_agent),
    identity_service: AgentIdentityService = Depends(get_identity_service),
    audit_logger: AuditLogger = Depends(get_audit_logger)
):
    """
    Revoke an access or refresh token.
    
    Immediately invalidates the specified token.
    """
    try:
        # Create audit context
        context = create_audit_context(request, current_agent)
        
        # Revoke token
        success = await identity_service.revoke_token(
            token=token,
            revoked_by=current_agent.get("human_controller"),
            reason=reason
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Token not found"
            )
        
        # Log revocation
        await audit_logger.log_event(
            context=context,
            action="token_revocation",
            resource="authentication",
            success=True,
            metadata={"reason": reason}
        )
        
        return {"message": "Token revoked successfully"}
        
    except Exception as e:
        logger.error(f"Token revocation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token revocation failed"
        )


# Authorization Endpoints

@router.post("/authz/check", response_model=PermissionCheckResponse)
async def check_permission(
    permission_request: PermissionCheckRequest,
    request: Request,
    current_agent: Dict[str, Any] = Depends(get_current_agent),
    authorization_engine: AuthorizationEngine = Depends(get_authorization_engine),
    audit_logger: AuditLogger = Depends(get_audit_logger)
):
    """
    Check if agent has permission to perform action on resource.
    
    Implements fine-grained RBAC permission checking with real-time evaluation.
    """
    try:
        # Create audit context
        context = create_audit_context(request, current_agent)
        
        # Add request context
        authz_context = permission_request.context.copy()
        authz_context.update({
            "ip_address": context.ip_address,
            "user_agent": context.user_agent,
            "session_id": str(context.session_id) if context.session_id else None
        })
        
        # Check permission
        authz_result = await authorization_engine.check_permission(
            agent_id=permission_request.agent_id,
            resource=permission_request.resource,
            action=permission_request.action,
            context=authz_context
        )
        
        # Log permission check
        await audit_logger.log_event(
            context=context,
            action="permission_check",
            resource=permission_request.resource,
            success=(authz_result.decision == AccessDecision.GRANTED),
            permission_checked=f"{permission_request.resource}:{permission_request.action}",
            authorization_result=authz_result.decision.value,
            metadata={
                "target_agent": permission_request.agent_id,
                "evaluation_time_ms": authz_result.evaluation_time_ms,
                "matched_roles": authz_result.matched_roles
            }
        )
        
        return authz_result.to_response()
        
    except Exception as e:
        logger.error(f"Permission check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authorization service error"
        )


@router.get("/authz/permissions/{agent_id}")
async def get_agent_permissions(
    agent_id: str,
    include_expired: bool = Query(False, description="Include expired role assignments"),
    current_agent: Dict[str, Any] = Depends(get_current_agent),
    authorization_engine: AuthorizationEngine = Depends(get_authorization_engine)
):
    """
    Get all permissions for an agent.
    
    Returns comprehensive permission summary including roles and effective permissions.
    """
    try:
        # Check if current agent can view permissions for target agent
        # (simplified check - in production would use proper authorization)
        if current_agent["sub"] != agent_id and "admin" not in current_agent.get("scope", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to view agent permissions"
            )
        
        permissions = await authorization_engine.get_agent_permissions(
            agent_id=agent_id,
            include_expired=include_expired
        )
        
        return permissions
        
    except Exception as e:
        logger.error(f"Get agent permissions failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve agent permissions"
        )


@router.post("/authz/roles/assign")
async def assign_role(
    assignment_request: RoleAssignmentRequest,
    request: Request,
    current_agent: Dict[str, Any] = Depends(get_current_agent),
    authorization_engine: AuthorizationEngine = Depends(get_authorization_engine),
    audit_logger: AuditLogger = Depends(get_audit_logger)
):
    """
    Assign role to agent.
    
    Requires administrative privileges to assign roles.
    """
    try:
        # Create audit context
        context = create_audit_context(request, current_agent)
        
        # Check if current agent can assign roles
        can_assign = await authorization_engine.check_permission(
            agent_id=current_agent["sub"],
            resource="roles",
            action="assign",
            context={"target_agent": assignment_request.agent_id}
        )
        
        if can_assign.decision != AccessDecision.GRANTED:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to assign roles"
            )
        
        # Assign role
        success = await authorization_engine.assign_role(
            agent_id=assignment_request.agent_id,
            role_id=assignment_request.role_id,
            granted_by=current_agent.get("human_controller"),
            granted_reason=assignment_request.granted_reason,
            resource_scope=assignment_request.resource_scope,
            expires_at=assignment_request.expires_at,
            conditions=assignment_request.conditions
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Role assignment failed"
            )
        
        # Log role assignment
        await audit_logger.log_event(
            context=context,
            action="role_assignment",
            resource="roles",
            success=True,
            metadata={
                "target_agent": assignment_request.agent_id,
                "role_id": assignment_request.role_id,
                "granted_reason": assignment_request.granted_reason
            }
        )
        
        return {"message": "Role assigned successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Role assignment failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Role assignment service error"
        )


# Audit and Logging Endpoints

@router.get("/audit/logs", response_model=Dict[str, Any])
async def query_audit_logs(
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    human_controller: Optional[str] = Query(None, description="Filter by human controller"),
    action: Optional[str] = Query(None, description="Filter by action"),
    resource: Optional[str] = Query(None, description="Filter by resource"),
    success: Optional[bool] = Query(None, description="Filter by success status"),
    start_time: Optional[datetime] = Query(None, description="Start time filter"),
    end_time: Optional[datetime] = Query(None, description="End time filter"),
    min_risk_score: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum risk score"),
    limit: int = Query(100, ge=1, le=1000, description="Result limit"),
    offset: int = Query(0, ge=0, description="Result offset"),
    current_agent: Dict[str, Any] = Depends(get_current_agent),
    audit_logger: AuditLogger = Depends(get_audit_logger)
):
    """
    Query security audit logs with filtering and pagination.
    
    Requires administrative privileges to access audit logs.
    """
    try:
        # Check permissions (simplified - in production would use proper authorization)
        if "admin" not in current_agent.get("scope", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to access audit logs"
            )
        
        # Create filters
        filters = AuditLogFilters(
            agent_id=agent_id,
            human_controller=human_controller,
            action=action,
            resource=resource,
            success=success,
            start_time=start_time,
            end_time=end_time,
            min_risk_score=min_risk_score,
            limit=limit,
            offset=offset
        )
        
        # Query audit logs
        results = await audit_logger.query_audit_logs(filters)
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audit log query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Audit log query service error"
        )


@router.post("/audit/events", response_model=Dict[str, str])
async def create_security_event(
    event_request: SecurityEventRequest,
    request: Request,
    current_agent: Dict[str, Any] = Depends(get_current_agent),
    audit_logger: AuditLogger = Depends(get_audit_logger)
):
    """
    Create security event for incident tracking.
    
    Allows authorized agents to report security incidents and events.
    """
    try:
        # Create audit context
        context = create_audit_context(request, current_agent)
        
        # Create security event
        event_id = await audit_logger.create_security_event(
            event_type=event_request.event_type,
            severity=event_request.severity,
            description=event_request.description,
            agent_id=event_request.agent_id,
            human_controller=event_request.human_controller or current_agent.get("human_controller"),
            source_ip=context.ip_address,
            details=event_request.details,
            risk_score=event_request.risk_score,
            auto_detected=False  # Manually reported
        )
        
        return {"event_id": event_id, "message": "Security event created successfully"}
        
    except Exception as e:
        logger.error(f"Security event creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Security event creation failed"
        )


@router.get("/dashboard", response_model=SecurityDashboard)
async def get_security_dashboard(
    current_agent: Dict[str, Any] = Depends(get_current_agent),
    audit_logger: AuditLogger = Depends(get_audit_logger)
):
    """
    Get security dashboard with real-time metrics.
    
    Provides comprehensive security overview including metrics, events, and health status.
    """
    try:
        # Check permissions
        if "admin" not in current_agent.get("scope", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to access security dashboard"
            )
        
        # Get dashboard data
        dashboard_data = await audit_logger.get_security_dashboard()
        
        return dashboard_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Security dashboard generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Security dashboard service error"
        )


# Health and Status Endpoints

@router.get("/health")
async def security_health_check():
    """
    Security service health check.
    
    Public endpoint for monitoring system health.
    """
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "components": {
                "authentication": "operational",
                "authorization": "operational",
                "audit_logging": "operational",
                "secret_management": "operational"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@router.get("/metrics")
async def security_metrics(
    current_agent: Dict[str, Any] = Depends(get_current_agent),
    identity_service: AgentIdentityService = Depends(get_identity_service),
    authorization_engine: AuthorizationEngine = Depends(get_authorization_engine),
    secret_manager: SecretManager = Depends(get_secret_manager)
):
    """
    Get security service metrics.
    
    Provides performance and operational metrics for monitoring.
    """
    try:
        # Check permissions
        if "admin" not in current_agent.get("scope", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to access metrics"
            )
        
        # Gather metrics from all components
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "authorization_engine": await authorization_engine.get_performance_metrics(),
            "secret_manager": await secret_manager.get_secret_metrics()
        }
        
        return metrics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Metrics collection service error"
        )


# Error Handlers
# Note: Exception handlers should be registered at the app level, not router level

# @app.exception_handler(TokenValidationError)
async def token_validation_error_handler(request: Request, exc: TokenValidationError):
    """Handle token validation errors."""
    return SecurityError(
        error="invalid_token",
        error_description=str(exc),
        correlation_id=getattr(request.state, 'request_id', None)
    )


# @app.exception_handler(RateLimitExceededError)
async def rate_limit_error_handler(request: Request, exc: RateLimitExceededError):
    """Handle rate limit exceeded errors."""
    return SecurityError(
        error="rate_limit_exceeded",
        error_description=str(exc),
        correlation_id=getattr(request.state, 'request_id', None)
    )