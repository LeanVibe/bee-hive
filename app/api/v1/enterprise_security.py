"""
Enterprise Security API Endpoints for LeanVibe Agent Hive 2.0.

Provides comprehensive enterprise-grade security management endpoints including:
- Enterprise authentication and authorization
- Security monitoring and threat management
- Compliance reporting and audit management
- Penetration testing and vulnerability management
- Security configuration and policy management
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import logging

from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field, validator
import httpx

from ...database import get_db_session
from ...models.agent import Agent
from ...core.enterprise_auth import (
    EnterpriseAuthenticationSystem, AuthenticatedUser, APIKey, 
    AuthenticationMethod, UserRole, OrganizationTier
)
from ...core.compliance_audit import (
    ComplianceAuditSystem, ComplianceFramework, AuditEventCategory, SeverityLevel
)
from ...core.security_monitoring import (
    SecurityMonitoringSystem, ThreatType, ThreatLevel, SecurityEvent
)
from ...core.penetration_testing import (
    PenetrationTestingFramework, PenTestType, TestSeverity
)
from ...core.security_audit import SecurityAuditSystem


logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/v1/security", tags=["Enterprise Security"])

# Security dependencies
security = HTTPBearer()


# Pydantic models for request/response
class SAMLAuthRequest(BaseModel):
    """SAML authentication request."""
    saml_assertion: str = Field(..., description="Base64 encoded SAML assertion")
    organization_id: str = Field(..., description="Organization ID")
    
    class Config:
        schema_extra = {
            "example": {
                "saml_assertion": "PHNhbWw6QXNzZXJ0aW9uLi4u",
                "organization_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        }


class OAuthAuthRequest(BaseModel):
    """OAuth authentication request."""
    oauth_token: str = Field(..., description="OAuth access token")
    provider: str = Field(..., description="OAuth provider name")
    organization_id: str = Field(..., description="Organization ID")
    
    class Config:
        schema_extra = {
            "example": {
                "oauth_token": "ya29.a0ARrdaM...",
                "provider": "google",
                "organization_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        }


class CreateAPIKeyRequest(BaseModel):
    """Create API key request."""
    name: str = Field(..., description="API key name", max_length=100)
    permissions: List[str] = Field(..., description="List of permissions")
    scopes: List[str] = Field(..., description="List of API scopes")
    expires_days: Optional[int] = Field(None, description="Days until expiration")
    rate_limit: int = Field(1000, description="Requests per hour limit")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Production API Key",
                "permissions": ["agent:read", "context:write", "workflow:execute"],
                "scopes": ["api", "agents", "workflows"],
                "expires_days": 365,
                "rate_limit": 5000
            }
        }


class SecurityEventRequest(BaseModel):
    """Security event monitoring request."""
    event_type: str = Field(..., description="Type of security event")
    source_ip: Optional[str] = Field(None, description="Source IP address")
    user_id: Optional[str] = Field(None, description="User ID")
    resource: Optional[str] = Field(None, description="Target resource")
    action: str = Field(..., description="Action performed")
    details: Dict[str, Any] = Field(default_factory=dict, description="Event details")
    
    class Config:
        schema_extra = {
            "example": {
                "event_type": "SUSPICIOUS_BEHAVIOR",
                "source_ip": "192.168.1.100",
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "resource": "/api/v1/agents",
                "action": "READ",
                "details": {"user_agent": "curl/7.68.0", "payload_size": 1024}
            }
        }


class ComplianceReportRequest(BaseModel):
    """Compliance report generation request."""
    framework: str = Field(..., description="Compliance framework")
    organization_id: str = Field(..., description="Organization ID")
    period_start: datetime = Field(..., description="Report period start")
    period_end: datetime = Field(..., description="Report period end")
    
    @validator('framework')
    def validate_framework(cls, v):
        valid_frameworks = [f.value for f in ComplianceFramework]
        if v not in valid_frameworks:
            raise ValueError(f"Invalid framework. Must be one of: {valid_frameworks}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "framework": "SOC2_TYPE2",
                "organization_id": "550e8400-e29b-41d4-a716-446655440000",
                "period_start": "2024-01-01T00:00:00Z",
                "period_end": "2024-12-31T23:59:59Z"
            }
        }


class PenTestRequest(BaseModel):
    """Penetration test request."""
    target_name: str = Field(..., description="Target name", max_length=100)
    target_type: str = Field(..., description="Target type")
    endpoint: str = Field(..., description="Target endpoint/URL")
    test_types: Optional[List[str]] = Field(None, description="Specific test types to run")
    credentials: Optional[Dict[str, Any]] = Field(None, description="Authentication credentials")
    scope: Optional[List[str]] = Field(None, description="Test scope")
    
    @validator('target_type')
    def validate_target_type(cls, v):
        valid_types = ["API", "WEB_APP", "SERVICE", "NETWORK"]
        if v not in valid_types:
            raise ValueError(f"Invalid target type. Must be one of: {valid_types}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "target_name": "Agent Hive API",
                "target_type": "API",
                "endpoint": "https://api.leanvibe.com",
                "test_types": ["API_SECURITY", "AUTHENTICATION", "AUTHORIZATION"],
                "scope": ["/api/v1/*"]
            }
        }


class AuthenticationResponse(BaseModel):
    """Authentication response."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiry in seconds")
    user: Dict[str, Any] = Field(..., description="User information")
    
    class Config:
        schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 28800,
                "user": {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "username": "john.doe",
                    "email": "john.doe@example.com",
                    "role": "DEVELOPER",
                    "organization": "Acme Corp"
                }
            }
        }


class APIKeyResponse(BaseModel):
    """API key creation response."""
    api_key: str = Field(..., description="Generated API key")
    key_id: str = Field(..., description="API key ID")
    expires_at: Optional[datetime] = Field(None, description="Expiry date")
    
    class Config:
        schema_extra = {
            "example": {
                "api_key": "lv_1234567890abcdef1234567890abcdef12345678",
                "key_id": "550e8400-e29b-41d4-a716-446655440000",
                "expires_at": "2025-01-01T00:00:00Z"
            }
        }


# Dependency to get enterprise auth system
async def get_enterprise_auth(db: AsyncSession = Depends(get_db_session)) -> EnterpriseAuthenticationSystem:
    """Get enterprise authentication system dependency."""
    # In production, these would be from configuration/environment
    from ...core.security_audit import create_security_audit_system
    from ...core.access_control import get_access_control_manager
    
    access_control = await get_access_control_manager(db)
    security_audit = await create_security_audit_system(db, access_control)
    
    jwt_secret = "your-super-secret-jwt-key"  # Use proper secret management
    encryption_key = b"your-32-byte-encryption-key-here"  # Use proper key management
    
    from ...core.enterprise_auth import create_enterprise_auth_system
    return await create_enterprise_auth_system(db, security_audit, jwt_secret, encryption_key)


# Dependency to get compliance audit system
async def get_compliance_audit(db: AsyncSession = Depends(get_db_session)) -> ComplianceAuditSystem:
    """Get compliance audit system dependency."""
    encryption_keys = {"default": b"your-32-byte-encryption-key-here"}
    
    from ...core.compliance_audit import create_compliance_audit_system
    return await create_compliance_audit_system(db, encryption_keys)


# Dependency to get security monitoring system
async def get_security_monitoring(
    db: AsyncSession = Depends(get_db_session),
    enterprise_auth: EnterpriseAuthenticationSystem = Depends(get_enterprise_auth),
    compliance_audit: ComplianceAuditSystem = Depends(get_compliance_audit)
) -> SecurityMonitoringSystem:
    """Get security monitoring system dependency."""
    from ...core.security_monitoring import create_security_monitoring_system
    return await create_security_monitoring_system(
        db, enterprise_auth.security_audit, compliance_audit
    )


# Dependency to get penetration testing framework
async def get_pentest_framework(
    db: AsyncSession = Depends(get_db_session),
    enterprise_auth: EnterpriseAuthenticationSystem = Depends(get_enterprise_auth),
    compliance_audit: ComplianceAuditSystem = Depends(get_compliance_audit),
    security_monitoring: SecurityMonitoringSystem = Depends(get_security_monitoring)
) -> PenetrationTestingFramework:
    """Get penetration testing framework dependency."""
    from ...core.penetration_testing import create_penetration_testing_framework
    return await create_penetration_testing_framework(
        db, enterprise_auth.security_audit, compliance_audit, security_monitoring
    )


# Dependency to get current authenticated user
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    enterprise_auth: EnterpriseAuthenticationSystem = Depends(get_enterprise_auth)
) -> AuthenticatedUser:
    """Get current authenticated user from JWT token."""
    try:
        token = credentials.credentials
        user = await enterprise_auth.validate_jwt_token(token)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        return user
        
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"}
        )


# Dependency to check admin permissions
async def require_admin(current_user: AuthenticatedUser = Depends(get_current_user)) -> AuthenticatedUser:
    """Require admin permissions."""
    if not current_user.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin permissions required"
        )
    return current_user


# Authentication endpoints
@router.post("/auth/saml", response_model=AuthenticationResponse)
async def authenticate_saml(
    request: SAMLAuthRequest,
    enterprise_auth: EnterpriseAuthenticationSystem = Depends(get_enterprise_auth)
):
    """Authenticate user via SAML assertion."""
    try:
        user = await enterprise_auth.authenticate_saml_user(
            request.saml_assertion,
            uuid.UUID(request.organization_id)
        )
        
        # Create JWT token
        access_token = await enterprise_auth.create_jwt_token(user)
        
        return AuthenticationResponse(
            access_token=access_token,
            expires_in=28800,  # 8 hours
            user=user.to_dict()
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"SAML authentication failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed"
        )


@router.post("/auth/oauth", response_model=AuthenticationResponse)
async def authenticate_oauth(
    request: OAuthAuthRequest,
    enterprise_auth: EnterpriseAuthenticationSystem = Depends(get_enterprise_auth)
):
    """Authenticate user via OAuth token."""
    try:
        user = await enterprise_auth.authenticate_oauth_user(
            request.oauth_token,
            request.provider,
            uuid.UUID(request.organization_id)
        )
        
        # Create JWT token
        access_token = await enterprise_auth.create_jwt_token(user)
        
        return AuthenticationResponse(
            access_token=access_token,
            expires_in=28800,  # 8 hours
            user=user.to_dict()
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"OAuth authentication failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed"
        )


@router.get("/auth/me")
async def get_current_user_info(current_user: AuthenticatedUser = Depends(get_current_user)):
    """Get current authenticated user information."""
    return current_user.to_dict()


# API Key management endpoints
@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    request: CreateAPIKeyRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    enterprise_auth: EnterpriseAuthenticationSystem = Depends(get_enterprise_auth)
):
    """Create new API key."""
    try:
        raw_key, api_key = await enterprise_auth.create_api_key(
            user=current_user,
            name=request.name,
            permissions=request.permissions,
            scopes=request.scopes,
            expires_days=request.expires_days,
            rate_limit=request.rate_limit
        )
        
        return APIKeyResponse(
            api_key=raw_key,
            key_id=str(api_key.id),
            expires_at=api_key.expires_at
        )
        
    except Exception as e:
        logger.error(f"API key creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key"
        )


@router.get("/api-keys")
async def list_api_keys(
    current_user: AuthenticatedUser = Depends(get_current_user),
    enterprise_auth: EnterpriseAuthenticationSystem = Depends(get_enterprise_auth)
):
    """List user's API keys."""
    try:
        # Get user's API keys (simplified - in production, query from database)
        user_keys = [
            key.to_dict() for key in enterprise_auth.api_keys.values()
            if key.user_id == current_user.id
        ]
        
        return {"api_keys": user_keys}
        
    except Exception as e:
        logger.error(f"API key listing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list API keys"
        )


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
    enterprise_auth: EnterpriseAuthenticationSystem = Depends(get_enterprise_auth)
):
    """Revoke API key."""
    try:
        key_uuid = uuid.UUID(key_id)
        
        # Find and revoke API key
        for key_hash, api_key in enterprise_auth.api_keys.items():
            if api_key.id == key_uuid and api_key.user_id == current_user.id:
                api_key.is_active = False
                return {"message": "API key revoked successfully"}
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid API key ID"
        )
    except Exception as e:
        logger.error(f"API key revocation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke API key"
        )


# Security monitoring endpoints
@router.post("/events")
async def submit_security_event(
    request: SecurityEventRequest,
    http_request: Request,
    current_user: AuthenticatedUser = Depends(get_current_user),
    security_monitoring: SecurityMonitoringSystem = Depends(get_security_monitoring)
):
    """Submit security event for monitoring."""
    try:
        # Extract request metadata
        source_ip = request.source_ip or http_request.client.host
        user_agent = http_request.headers.get("user-agent")
        
        # Create event data
        event_data = {
            "event_type": request.event_type,
            "source_ip": source_ip,
            "user_id": request.user_id or str(current_user.id),
            "user_agent": user_agent,
            "resource": request.resource,
            "action": request.action,
            "details": request.details,
            "timestamp": datetime.utcnow().isoformat(),
            "severity": "MEDIUM"  # Default severity
        }
        
        # Process security events
        processed_events = await security_monitoring.monitor_security_events([event_data])
        
        return {
            "message": "Security event processed",
            "event_id": str(processed_events[0].id) if processed_events else None,
            "risk_score": processed_events[0].risk_score if processed_events else 0.0
        }
        
    except Exception as e:
        logger.error(f"Security event processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process security event"
        )


@router.get("/dashboard")
async def get_security_dashboard(
    current_user: AuthenticatedUser = Depends(get_current_user),
    security_monitoring: SecurityMonitoringSystem = Depends(get_security_monitoring)
):
    """Get real-time security dashboard."""
    try:
        # Check if user has security monitoring permissions
        if not current_user.has_permission("security:read"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Security monitoring permissions required"
            )
        
        dashboard = await security_monitoring.generate_security_dashboard(
            organization_id=current_user.organization_id if not current_user.is_admin() else None
        )
        
        return dashboard
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Security dashboard generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate security dashboard"
        )


@router.get("/threats")
async def list_security_threats(
    current_user: AuthenticatedUser = Depends(get_current_user),
    security_monitoring: SecurityMonitoringSystem = Depends(get_security_monitoring),
    limit: int = 50,
    severity: Optional[str] = None
):
    """List recent security threats."""
    try:
        # Check permissions
        if not current_user.has_permission("security:read"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Security monitoring permissions required"
            )
        
        # Get recent security events
        recent_events = [
            event.to_dict() for event in security_monitoring.event_buffer
            if event.timestamp >= datetime.utcnow() - timedelta(hours=24)
        ]
        
        # Filter by severity if specified
        if severity:
            recent_events = [
                event for event in recent_events
                if event.get("severity") == severity
            ]
        
        # Limit results
        recent_events = recent_events[:limit]
        
        return {
            "threats": recent_events,
            "total": len(recent_events),
            "period": "24 hours"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Threat listing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list security threats"
        )


@router.get("/incidents")
async def list_security_incidents(
    current_user: AuthenticatedUser = Depends(get_current_user),
    security_monitoring: SecurityMonitoringSystem = Depends(get_security_monitoring),
    limit: int = 20,
    status_filter: Optional[str] = None
):
    """List security incidents."""
    try:
        # Check permissions
        if not current_user.has_permission("security:read"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Security monitoring permissions required"
            )
        
        # Get active incidents
        incidents = [
            incident.to_dict() for incident in security_monitoring.active_incidents.values()
        ]
        
        # Filter by status if specified
        if status_filter:
            incidents = [
                incident for incident in incidents
                if incident.get("status") == status_filter
            ]
        
        # Limit results
        incidents = incidents[:limit]
        
        return {
            "incidents": incidents,
            "total": len(incidents)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Incident listing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list security incidents"
        )


# Compliance reporting endpoints
@router.post("/compliance/reports")
async def generate_compliance_report(
    request: ComplianceReportRequest,
    current_user: AuthenticatedUser = Depends(require_admin),
    compliance_audit: ComplianceAuditSystem = Depends(get_compliance_audit)
):
    """Generate compliance report."""
    try:
        # Parse framework
        framework = ComplianceFramework(request.framework)
        
        # Generate report
        report = await compliance_audit.generate_compliance_report(
            framework=framework,
            organization_id=uuid.UUID(request.organization_id),
            period_start=request.period_start,
            period_end=request.period_end,
            generated_by=current_user.id
        )
        
        return report.to_dict()
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Compliance report generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate compliance report"
        )


@router.get("/compliance/frameworks")
async def list_compliance_frameworks(
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """List supported compliance frameworks."""
    try:
        # Check permissions
        if not current_user.has_permission("compliance:read"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Compliance permissions required"
            )
        
        frameworks = [
            {
                "name": framework.value,
                "description": f"{framework.value.replace('_', ' ').title()} compliance framework"
            }
            for framework in ComplianceFramework
        ]
        
        return {"frameworks": frameworks}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Framework listing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list compliance frameworks"
        )


@router.post("/compliance/scan")
async def run_compliance_scan(
    framework: str,
    organization_id: Optional[str] = None,
    current_user: AuthenticatedUser = Depends(require_admin),
    compliance_audit: ComplianceAuditSystem = Depends(get_compliance_audit)
):
    """Run automated compliance scan."""
    try:
        # Parse framework
        compliance_framework = ComplianceFramework(framework)
        
        # Parse organization ID
        org_id = uuid.UUID(organization_id) if organization_id else None
        
        # Run compliance checks
        results = await compliance_audit.automate_compliance_checks(
            framework=compliance_framework,
            organization_id=org_id
        )
        
        return results
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Compliance scan failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run compliance scan"
        )


# Penetration testing endpoints
@router.post("/pentest")
async def create_penetration_test(
    request: PenTestRequest,
    current_user: AuthenticatedUser = Depends(require_admin),
    pentest_framework: PenetrationTestingFramework = Depends(get_pentest_framework)
):
    """Create and execute penetration test."""
    try:
        # Create test target
        target = await pentest_framework.create_test_target(
            name=request.target_name,
            target_type=request.target_type,
            endpoint=request.endpoint,
            credentials=request.credentials,
            scope=request.scope or []
        )
        
        # Parse test types if specified
        test_types = None
        if request.test_types:
            test_types = [PenTestType(test_type) for test_type in request.test_types]
        
        # Execute test suite
        results = await pentest_framework.execute_security_test_suite(
            target_id=target.id,
            test_types=test_types
        )
        
        return {
            "test_id": str(results.id),
            "target": target.to_dict(),
            "status": "COMPLETED",
            "summary": results.calculate_summary()
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Penetration test execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute penetration test"
        )


@router.get("/pentest/{test_id}/results")
async def get_pentest_results(
    test_id: str,
    current_user: AuthenticatedUser = Depends(require_admin),
    pentest_framework: PenetrationTestingFramework = Depends(get_pentest_framework)
):
    """Get penetration test results."""
    try:
        test_uuid = uuid.UUID(test_id)
        
        # Get test results
        test_results = pentest_framework.active_tests.get(test_uuid)
        
        if not test_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Penetration test results not found"
            )
        
        # Validate results
        security_score = await pentest_framework.validate_penetration_test_results(
            test_results
        )
        
        return {
            "test_results": test_results.calculate_summary(),
            "security_score": security_score.to_dict(),
            "detailed_findings": test_results.detailed_findings,
            "remediation_roadmap": test_results.remediation_roadmap
        }
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid test ID"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pentest results retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve penetration test results"
        )


@router.get("/pentest/{test_id}/report")
async def generate_pentest_report(
    test_id: str,
    format_type: str = "JSON",
    current_user: AuthenticatedUser = Depends(require_admin),
    pentest_framework: PenetrationTestingFramework = Depends(get_pentest_framework)
):
    """Generate penetration test report."""
    try:
        test_uuid = uuid.UUID(test_id)
        
        # Get test results
        test_results = pentest_framework.active_tests.get(test_uuid)
        
        if not test_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Penetration test results not found"
            )
        
        # Validate results and get security score
        security_score = await pentest_framework.validate_penetration_test_results(
            test_results
        )
        
        # Generate comprehensive report
        report = await pentest_framework.generate_security_validation_report(
            test_results=test_results,
            security_score=security_score,
            format_type=format_type
        )
        
        return report
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid test ID"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pentest report generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate penetration test report"
        )


# Audit and logging endpoints
@router.get("/audit/events")
async def get_audit_events(
    current_user: AuthenticatedUser = Depends(require_admin),
    compliance_audit: ComplianceAuditSystem = Depends(get_compliance_audit),
    limit: int = 100,
    event_type: Optional[str] = None,
    hours_back: int = 24
):
    """Get recent audit events."""
    try:
        # Calculate time range
        start_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        # Get audit events (simplified - in production, query from persistent storage)
        events = []
        for event in compliance_audit.audit_buffer:
            if event.timestamp >= start_time:
                if not event_type or event.event_type == event_type:
                    events.append(event.to_dict())
        
        # Limit results
        events = events[:limit]
        
        return {
            "events": events,
            "total": len(events),
            "period_hours": hours_back
        }
        
    except Exception as e:
        logger.error(f"Audit event retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve audit events"
        )


@router.get("/health")
async def security_health_check(
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """Get security system health status."""
    try:
        # Check permissions
        if not current_user.has_permission("security:read"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Security monitoring permissions required"
            )
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "systems": {
                "enterprise_auth": {"status": "operational"},
                "security_monitoring": {"status": "operational"},
                "compliance_audit": {"status": "operational"},
                "penetration_testing": {"status": "operational"}
            },
            "version": "2.0.0"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health check failed"
        )