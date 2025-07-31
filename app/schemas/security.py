"""
Security & Authentication Schemas for API validation.

Pydantic models for request/response validation, JWT token structures,
and security audit data serialization.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, validator, EmailStr
from pydantic import IPvAnyAddress


class AgentStatusEnum(str, Enum):
    """Agent status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    REVOKED = "revoked"


class RoleScopeEnum(str, Enum):
    """Role scope enumeration."""
    GLOBAL = "global"
    SESSION = "session"
    CONTEXT = "context"
    RESOURCE = "resource"


class SecurityEventSeverityEnum(str, Enum):
    """Security event severity enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TokenTypeEnum(str, Enum):
    """Token type enumeration."""
    ACCESS = "access"
    REFRESH = "refresh"


# Request Schemas

class AgentTokenRequest(BaseModel):
    """Request schema for agent token generation."""
    agent_id: str = Field(..., description="Agent identifier")
    human_controller: str = Field(..., description="Human controller email or identifier")
    requested_scopes: List[str] = Field(default=[], description="Requested access scopes")
    client_credentials: Optional[Dict[str, str]] = Field(None, description="OAuth client credentials")
    
    @validator('agent_id')
    def validate_agent_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Agent ID cannot be empty')
        return v.strip()
    
    @validator('human_controller')
    def validate_human_controller(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Human controller cannot be empty')
        return v.strip()


class TokenRefreshRequest(BaseModel):
    """Request schema for token refresh."""
    refresh_token: str = Field(..., description="Refresh token")
    requested_scopes: Optional[List[str]] = Field(None, description="Optional scope changes")


class PermissionCheckRequest(BaseModel):
    """Request schema for permission checking."""
    agent_id: str = Field(..., description="Agent identifier")
    resource: str = Field(..., description="Resource being accessed")
    action: str = Field(..., description="Action being performed")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Additional context")
    
    class Config:
        schema_extra = {
            "example": {
                "agent_id": "agent-001",
                "resource": "github/repositories/myorg/myrepo",
                "action": "write",
                "context": {
                    "ip_address": "10.0.0.1",
                    "session_id": "session-123"
                }
            }
        }


class AuditLogRequest(BaseModel):
    """Request schema for audit logging."""
    agent_id: Optional[str] = Field(None, description="Agent identifier")
    human_controller: str = Field(..., description="Human controller")
    action: str = Field(..., description="Action performed")
    resource: Optional[str] = Field(None, description="Resource accessed")
    resource_id: Optional[str] = Field(None, description="Specific resource ID")
    method: Optional[str] = Field(None, description="HTTP method")
    endpoint: Optional[str] = Field(None, description="API endpoint")
    request_data: Optional[Dict[str, Any]] = Field(None, description="Request payload")
    response_data: Optional[Dict[str, Any]] = Field(None, description="Response payload")
    success: bool = Field(..., description="Operation success status")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    duration_ms: Optional[int] = Field(None, description="Operation duration in milliseconds")
    
    @validator('duration_ms')
    def validate_duration(cls, v):
        if v is not None and v < 0:
            raise ValueError('Duration cannot be negative')
        return v


class RoleAssignmentRequest(BaseModel):
    """Request schema for role assignment."""
    agent_id: str = Field(..., description="Agent identifier")
    role_id: str = Field(..., description="Role identifier")
    granted_by: str = Field(..., description="Who granted the role")
    granted_reason: Optional[str] = Field(None, description="Reason for granting")
    resource_scope: Optional[str] = Field(None, description="Specific resource scope")
    expires_at: Optional[datetime] = Field(None, description="Expiration time")
    conditions: Optional[Dict[str, Any]] = Field(default={}, description="Access conditions")


class SecurityEventRequest(BaseModel):
    """Request schema for security event creation."""
    event_type: str = Field(..., description="Type of security event")
    severity: SecurityEventSeverityEnum = Field(..., description="Event severity")
    agent_id: Optional[str] = Field(None, description="Associated agent ID")
    human_controller: Optional[str] = Field(None, description="Associated human controller")
    description: str = Field(..., description="Event description")
    details: Optional[Dict[str, Any]] = Field(default={}, description="Additional event details")
    risk_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Risk score (0.0-1.0)")
    
    @validator('risk_score')
    def validate_risk_score(cls, v):
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError('Risk score must be between 0.0 and 1.0')
        return v


# Response Schemas

class TokenResponse(BaseModel):
    """Response schema for token generation."""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: Optional[str] = Field(None, description="Refresh token")
    token_type: str = Field(default="Bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")
    scope: List[str] = Field(default=[], description="Granted scopes")
    
    class Config:
        schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
                "refresh_token": "def456...",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": ["read:files", "write:github"]
            }
        }


class PermissionCheckResponse(BaseModel):
    """Response schema for permission checking."""
    allowed: bool = Field(..., description="Whether access is allowed")
    reason: str = Field(..., description="Reason for the decision")
    effective_permissions: Optional[Dict[str, Any]] = Field(None, description="Effective permissions")
    conditions_met: Optional[bool] = Field(None, description="Whether conditions were met")
    
    class Config:
        schema_extra = {
            "example": {
                "allowed": True,
                "reason": "Agent has write access to this repository",
                "effective_permissions": {
                    "role": "developer",
                    "actions": ["read", "write"],
                    "resources": ["github/repositories/myorg/*"]
                },
                "conditions_met": True
            }
        }


class AgentIdentityResponse(BaseModel):
    """Response schema for agent identity."""
    id: str = Field(..., description="Agent identity ID")
    agent_name: str = Field(..., description="Agent name")
    human_controller: str = Field(..., description="Human controller")
    oauth_client_id: Optional[str] = Field(None, description="OAuth client ID")
    scopes: List[str] = Field(default=[], description="Available scopes")
    status: AgentStatusEnum = Field(..., description="Agent status")
    rate_limit_per_minute: int = Field(..., description="Rate limit per minute")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_used: Optional[datetime] = Field(None, description="Last usage timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "agent_name": "research-agent-001",
                "human_controller": "user@company.com",
                "oauth_client_id": "client_abc123",
                "scopes": ["read:files", "write:github"],
                "status": "active",
                "rate_limit_per_minute": 10,
                "created_at": "2024-01-01T12:00:00Z",
                "last_used": "2024-01-01T15:30:00Z"
            }
        }


class AgentRoleResponse(BaseModel):
    """Response schema for agent role."""
    id: str = Field(..., description="Role ID")
    role_name: str = Field(..., description="Role name")
    display_name: Optional[str] = Field(None, description="Display name")
    description: Optional[str] = Field(None, description="Role description")
    scope: RoleScopeEnum = Field(..., description="Role scope")
    permissions: Dict[str, Any] = Field(..., description="Role permissions")
    resource_patterns: List[str] = Field(default=[], description="Resource patterns")
    max_access_level: str = Field(..., description="Maximum access level")
    can_delegate: bool = Field(..., description="Can delegate permissions")
    is_system_role: bool = Field(..., description="Is system role")
    created_at: datetime = Field(..., description="Creation timestamp")


class SecurityAuditLogResponse(BaseModel):
    """Response schema for security audit log entry."""
    id: str = Field(..., description="Log entry ID")
    agent_id: Optional[str] = Field(None, description="Agent ID")
    human_controller: str = Field(..., description="Human controller")
    action: str = Field(..., description="Action performed")
    resource: Optional[str] = Field(None, description="Resource accessed")
    success: bool = Field(..., description="Operation success")
    timestamp: datetime = Field(..., description="Event timestamp")
    duration_ms: Optional[int] = Field(None, description="Duration in milliseconds")
    ip_address: Optional[str] = Field(None, description="Source IP address")
    risk_score: Optional[float] = Field(None, description="Calculated risk score")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "log-123e4567-e89b-12d3-a456-426614174000",
                "agent_id": "agent-001",
                "human_controller": "user@company.com",
                "action": "read_file",
                "resource": "files/workspace/document.txt",
                "success": True,
                "timestamp": "2024-01-01T15:30:00Z",
                "duration_ms": 150,
                "ip_address": "10.0.0.1",
                "risk_score": 0.1
            }
        }


class SecurityEventResponse(BaseModel):
    """Response schema for security event."""
    id: str = Field(..., description="Event ID")
    event_type: str = Field(..., description="Event type")
    severity: SecurityEventSeverityEnum = Field(..., description="Event severity")
    agent_id: Optional[str] = Field(None, description="Associated agent ID")
    description: str = Field(..., description="Event description")
    risk_score: Optional[float] = Field(None, description="Risk score")
    is_resolved: bool = Field(..., description="Resolution status")
    timestamp: datetime = Field(..., description="Event timestamp")
    priority_score: float = Field(..., description="Calculated priority score")


# Dashboard and Analytics Schemas

class SecurityMetrics(BaseModel):
    """Security metrics for dashboard."""
    total_agents: int = Field(..., description="Total number of agents")
    active_agents: int = Field(..., description="Active agents")
    total_events_24h: int = Field(..., description="Total events in last 24 hours")
    failed_events_24h: int = Field(..., description="Failed events in last 24 hours")
    security_events_24h: int = Field(..., description="Security events in last 24 hours")
    avg_risk_score: float = Field(..., description="Average risk score")
    top_risk_agents: List[Dict[str, Any]] = Field(default=[], description="Top risk agents")


class SecurityDashboard(BaseModel):
    """Security dashboard response."""
    timestamp: datetime = Field(..., description="Dashboard generation timestamp")
    metrics: SecurityMetrics = Field(..., description="Security metrics")
    recent_events: List[SecurityEventResponse] = Field(default=[], description="Recent security events")
    threat_summary: Dict[str, int] = Field(default={}, description="Threat level summary")
    
    class Config:
        schema_extra = {
            "example": {
                "timestamp": "2024-01-01T16:00:00Z",
                "metrics": {
                    "total_agents": 25,
                    "active_agents": 20,
                    "total_events_24h": 1500,
                    "failed_events_24h": 45,
                    "security_events_24h": 3,
                    "avg_risk_score": 0.15,
                    "top_risk_agents": [
                        {"agent_id": "agent-001", "risk_score": 0.65},
                        {"agent_id": "agent-007", "risk_score": 0.42}
                    ]
                },
                "recent_events": [],
                "threat_summary": {
                    "low": 1,
                    "medium": 1,
                    "high": 1,
                    "critical": 0
                }
            }
        }


# JWT Token Payload Schema

class JWTTokenPayload(BaseModel):
    """JWT token payload structure."""
    iss: str = Field(..., description="Token issuer")
    sub: str = Field(..., description="Subject (agent ID)")
    aud: str = Field(..., description="Audience")
    exp: int = Field(..., description="Expiration timestamp")
    iat: int = Field(..., description="Issued at timestamp")
    jti: str = Field(..., description="JWT ID")
    scope: List[str] = Field(default=[], description="Token scopes")
    human_controller: str = Field(..., description="Human controller")
    agent_name: str = Field(..., description="Agent name")
    role_ids: List[str] = Field(default=[], description="Assigned role IDs")
    
    class Config:
        schema_extra = {
            "example": {
                "iss": "leanvibe-agent-hive",
                "sub": "agent-001",
                "aud": "leanvibe-api",
                "exp": 1640995200,
                "iat": 1640991600,
                "jti": "token-123e4567",
                "scope": ["read:files", "write:github"],
                "human_controller": "user@company.com",
                "agent_name": "research-agent-001",
                "role_ids": ["role-developer", "role-reader"]
            }
        }


# Query and Filter Schemas

class AuditLogFilters(BaseModel):
    """Filters for audit log queries."""
    agent_id: Optional[str] = Field(None, description="Filter by agent ID")
    human_controller: Optional[str] = Field(None, description="Filter by human controller")
    action: Optional[str] = Field(None, description="Filter by action")
    resource: Optional[str] = Field(None, description="Filter by resource")
    success: Optional[bool] = Field(None, description="Filter by success status")
    start_time: Optional[datetime] = Field(None, description="Start time filter")
    end_time: Optional[datetime] = Field(None, description="End time filter")
    min_risk_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum risk score")
    limit: int = Field(default=100, ge=1, le=1000, description="Result limit")
    offset: int = Field(default=0, ge=0, description="Result offset")


class SecurityEventFilters(BaseModel):
    """Filters for security event queries."""
    event_type: Optional[str] = Field(None, description="Filter by event type")
    severity: Optional[SecurityEventSeverityEnum] = Field(None, description="Filter by severity")
    agent_id: Optional[str] = Field(None, description="Filter by agent ID")
    is_resolved: Optional[bool] = Field(None, description="Filter by resolution status")
    start_time: Optional[datetime] = Field(None, description="Start time filter")
    end_time: Optional[datetime] = Field(None, description="End time filter")
    min_risk_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum risk score")
    limit: int = Field(default=100, ge=1, le=1000, description="Result limit")
    offset: int = Field(default=0, ge=0, description="Result offset")


# Error Response Schemas

class SecurityError(BaseModel):
    """Security error response."""
    error: str = Field(..., description="Error type")
    error_description: str = Field(..., description="Error description")
    error_code: Optional[str] = Field(None, description="Specific error code")
    correlation_id: Optional[str] = Field(None, description="Request correlation ID")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "access_denied",
                "error_description": "Agent does not have permission to access this resource",
                "error_code": "AUTH_001",
                "correlation_id": "req-123e4567"
            }
        }


class ValidationError(BaseModel):
    """Validation error response."""
    error: str = Field(default="validation_error", description="Error type")
    message: str = Field(..., description="Error message")
    details: List[Dict[str, Any]] = Field(default=[], description="Validation error details")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "validation_error",
                "message": "Request validation failed",
                "details": [
                    {
                        "field": "agent_id",
                        "message": "Agent ID cannot be empty",
                        "type": "value_error"
                    }
                ]
            }
        }


# OAuth 2.0/OIDC Schemas

class OAuthProviderType(str, Enum):
    """OAuth provider types."""
    GOOGLE = "google"
    GITHUB = "github"
    MICROSOFT = "microsoft"
    AZURE_AD = "azure_ad"
    CUSTOM_OIDC = "custom_oidc"


class OAuthProviderConfig(BaseModel):
    """OAuth provider configuration schema."""
    provider_name: str = Field(..., description="Unique provider name")
    provider_type: OAuthProviderType = Field(..., description="Provider type")
    client_id: str = Field(..., description="OAuth client ID")
    client_secret: str = Field(..., description="OAuth client secret")
    scopes: List[str] = Field(default=[], description="Default scopes")
    tenant_id: Optional[str] = Field(None, description="Tenant ID for Azure AD")
    domain_hint: Optional[str] = Field(None, description="Domain hint for authentication")
    redirect_uri: Optional[str] = Field(None, description="Custom redirect URI")
    
    # Custom endpoints for OIDC providers
    authorization_endpoint: Optional[str] = Field(None, description="Custom authorization endpoint")
    token_endpoint: Optional[str] = Field(None, description="Custom token endpoint")
    userinfo_endpoint: Optional[str] = Field(None, description="Custom userinfo endpoint")
    jwks_uri: Optional[str] = Field(None, description="JWKS endpoint")
    issuer: Optional[str] = Field(None, description="OIDC issuer")
    
    class Config:
        schema_extra = {
            "example": {
                "provider_name": "company_google",
                "provider_type": "google",
                "client_id": "123456789.apps.googleusercontent.com",
                "client_secret": "GOCSPX-abcdef123456",
                "scopes": ["openid", "email", "profile"],
                "domain_hint": "company.com"
            }
        }


class OAuthAuthorizationRequest(BaseModel):
    """OAuth authorization request schema."""
    provider_name: str = Field(..., description="OAuth provider name")
    redirect_uri: Optional[str] = Field(None, description="Custom redirect URI")
    scopes: Optional[List[str]] = Field(None, description="Custom scopes")
    state_data: Optional[Dict[str, Any]] = Field(None, description="Additional state data")
    
    class Config:
        schema_extra = {
            "example": {
                "provider_name": "company_google",
                "scopes": ["openid", "email", "profile"],
                "state_data": {"return_url": "/dashboard"}
            }
        }


class OAuthCallbackRequest(BaseModel):
    """OAuth callback request schema."""
    code: str = Field(..., description="Authorization code")
    state: str = Field(..., description="State parameter")
    error: Optional[str] = Field(None, description="Error code if authorization failed")
    error_description: Optional[str] = Field(None, description="Error description")


class OAuthTokenResponse(BaseModel):
    """OAuth token response schema."""
    access_token: str = Field(..., description="Access token")
    token_type: str = Field(default="Bearer", description="Token type")
    expires_in: Optional[int] = Field(None, description="Token expiration in seconds")
    refresh_token: Optional[str] = Field(None, description="Refresh token")
    id_token: Optional[str] = Field(None, description="OIDC ID token")
    scope: Optional[str] = Field(None, description="Granted scopes")
    
    class Config:
        schema_extra = {
            "example": {
                "access_token": "ya29.a0ARrdaM...",
                "token_type": "Bearer",
                "expires_in": 3599,
                "refresh_token": "1//04rYrP9...",
                "id_token": "eyJhbGciOiJSUzI1NiIs...",
                "scope": "openid email profile"
            }
        }


class UserProfile(BaseModel):
    """User profile from OAuth provider."""
    user_id: str = Field(..., description="Unique user ID from provider")
    email: str = Field(..., description="User email address")
    name: str = Field(..., description="User display name")
    avatar_url: Optional[str] = Field(None, description="User avatar URL")
    provider: str = Field(..., description="OAuth provider name")
    raw_data: Optional[Dict[str, Any]] = Field(None, description="Raw user data from provider")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "123456789",
                "email": "user@company.com",
                "name": "John Doe",
                "avatar_url": "https://lh3.googleusercontent.com/a/...",
                "provider": "google",
                "raw_data": {
                    "sub": "123456789",
                    "email": "user@company.com",
                    "email_verified": True,
                    "name": "John Doe",
                    "picture": "https://lh3.googleusercontent.com/a/..."
                }
            }
        }


class OAuthAuthorizationResponse(BaseModel):
    """OAuth authorization initiation response."""
    authorization_url: str = Field(..., description="Authorization URL to redirect to")
    session_id: str = Field(..., description="OAuth session ID")
    provider_name: str = Field(..., description="Provider name")
    expires_at: datetime = Field(..., description="Session expiration time")
    
    class Config:
        schema_extra = {
            "example": {
                "authorization_url": "https://accounts.google.com/o/oauth2/v2/auth?client_id=...",
                "session_id": "oauth_sess_123e4567",
                "provider_name": "company_google",
                "expires_at": "2024-01-01T12:15:00Z"
            }
        }


class OAuthCallbackResponse(BaseModel):
    """OAuth callback completion response."""
    success: bool = Field(..., description="Whether authorization was successful")
    user_profile: Optional[UserProfile] = Field(None, description="User profile if successful")
    access_token: Optional[str] = Field(None, description="Access token if successful")
    expires_in: Optional[int] = Field(None, description="Token expiration in seconds")
    error: Optional[str] = Field(None, description="Error message if failed")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "user_profile": {
                    "user_id": "123456789",
                    "email": "user@company.com",
                    "name": "John Doe",
                    "avatar_url": "https://lh3.googleusercontent.com/a/...",
                    "provider": "google"
                },
                "access_token": "leanvibe_jwt_token_here",
                "expires_in": 3600
            }
        }


class OAuthProviderListResponse(BaseModel):
    """List of configured OAuth providers."""
    providers: List[Dict[str, Any]] = Field(..., description="List of configured providers")
    
    class Config:
        schema_extra = {
            "example": {
                "providers": [
                    {
                        "name": "company_google",
                        "type": "google",
                        "scopes": ["openid", "email", "profile"],
                        "supports_refresh": True,
                        "supports_pkce": True
                    },
                    {
                        "name": "github_enterprise",
                        "type": "github",
                        "scopes": ["user:email", "read:user"],
                        "supports_refresh": True,
                        "supports_pkce": True
                    }
                ]
            }
        }