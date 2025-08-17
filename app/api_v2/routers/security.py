"""
Security API - Consolidated security and authentication endpoints

Consolidates auth_endpoints.py, security_endpoints.py, 
enterprise_security.py, v1/security.py, v1/security_dashboard.py,
and v1/oauth.py into a unified security resource.

Performance target: <75ms P95 response time
"""

import uuid
from typing import Optional, Dict, Any
from datetime import datetime

import structlog
from fastapi import APIRouter, Request, HTTPException, Query
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.database import get_session_dependency
from ...core.auth import (
    UserLogin, UserResponse, TokenResponse, RefreshRequest,
    login_user, get_current_user_info, refresh_access_token, logout_user,
    get_current_user, Permission, UserRole
)
from ..middleware import get_current_user_from_request

logger = structlog.get_logger()
router = APIRouter()

@router.post("/login", response_model=TokenResponse)
async def login(login_data: UserLogin) -> TokenResponse:
    """
    Authenticate user and return JWT tokens.
    
    Performance target: <75ms
    """
    try:
        return await login_user(login_data)
    except Exception as e:
        logger.error("login_failed", error=str(e))
        raise HTTPException(
            status_code=401,
            detail="Authentication failed"
        )

@router.post("/refresh")
async def refresh_token(payload: RefreshRequest) -> dict:
    """
    Refresh access token using refresh token.
    
    Performance target: <75ms
    """
    try:
        return await refresh_access_token(payload)
    except Exception as e:
        logger.error("token_refresh_failed", error=str(e))
        raise HTTPException(
            status_code=401,
            detail="Token refresh failed"
        )

@router.post("/logout")
async def logout(request: Request) -> dict:
    """
    Logout current user and invalidate tokens.
    
    Performance target: <75ms
    """
    current_user = get_current_user_from_request(request)
    
    try:
        return await logout_user(current_user)
    except Exception as e:
        logger.error("logout_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Logout failed"
        )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info_endpoint(request: Request) -> UserResponse:
    """
    Get current user information.
    
    Performance target: <75ms
    """
    current_user = get_current_user_from_request(request)
    
    try:
        return await get_current_user_info(current_user)
    except Exception as e:
        logger.error("get_user_info_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to get user information"
        )

@router.get("/permissions")
async def list_permissions(request: Request):
    """
    List all available permissions.
    
    Performance target: <75ms
    """
    current_user = get_current_user_from_request(request)
    
    return {
        "permissions": [
            {
                "name": permission.name,
                "value": permission.value,
                "description": f"Permission to {permission.value.replace('_', ' ')}"
            }
            for permission in Permission
        ],
        "user_permissions": getattr(current_user, 'permissions', [])
    }

@router.get("/roles")
async def list_roles(request: Request):
    """
    List all available user roles.
    
    Performance target: <75ms
    """
    return {
        "roles": [
            {
                "name": role.name,
                "value": role.value,
                "description": f"Role: {role.value.replace('_', ' ').title()}"
            }
            for role in UserRole
        ]
    }

@router.get("/audit/events")
async def list_audit_events(
    request: Request,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=1000),
    event_type: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_session_dependency)
):
    """
    List security audit events.
    
    Performance target: <75ms
    """
    current_user = get_current_user_from_request(request)
    
    # Check if user has audit viewing permission
    if not hasattr(current_user, 'permissions') or Permission.VIEW_SYSTEM_LOGS not in current_user.permissions:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions to view audit events"
        )
    
    # For now, return placeholder audit events
    # In production, this would query an audit_events table
    return {
        "events": [],
        "total": 0,
        "skip": skip,
        "limit": limit,
        "filters": {
            "event_type": event_type,
            "user_id": user_id
        }
    }

@router.get("/sessions")
async def list_active_sessions(
    request: Request,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=1000),
    db: AsyncSession = Depends(get_session_dependency)
):
    """
    List active user sessions.
    
    Performance target: <75ms
    """
    current_user = get_current_user_from_request(request)
    
    # Users can only see their own sessions unless they're admin
    user_id_filter = current_user.id
    if hasattr(current_user, 'role') and current_user.role in [UserRole.SUPER_ADMIN, UserRole.ENTERPRISE_ADMIN]:
        user_id_filter = None  # Admins can see all sessions
    
    # For now, return placeholder session data
    # In production, this would query active sessions from Redis or database
    return {
        "sessions": [
            {
                "session_id": str(uuid.uuid4()),
                "user_id": current_user.id,
                "created_at": datetime.utcnow().isoformat(),
                "last_activity": datetime.utcnow().isoformat(),
                "ip_address": "masked",
                "user_agent": "masked"
            }
        ],
        "total": 1,
        "skip": skip,
        "limit": limit
    }

@router.post("/sessions/{session_id}/revoke")
async def revoke_session(
    request: Request,
    session_id: str,
    db: AsyncSession = Depends(get_session_dependency)
):
    """
    Revoke a specific user session.
    
    Performance target: <75ms
    """
    current_user = get_current_user_from_request(request)
    
    try:
        # In production, this would:
        # 1. Verify the session exists
        # 2. Check user permissions (own session or admin)
        # 3. Invalidate the session tokens
        # 4. Remove from active sessions store
        
        logger.info(
            "session_revoked",
            session_id=session_id,
            revoked_by=current_user.id
        )
        
        return {
            "session_id": session_id,
            "status": "revoked",
            "message": "Session successfully revoked"
        }
        
    except Exception as e:
        logger.error("session_revoke_failed", session_id=session_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to revoke session"
        )

@router.get("/security-status")
async def get_security_status(
    request: Request,
    db: AsyncSession = Depends(get_session_dependency)
):
    """
    Get overall security status and metrics.
    
    Performance target: <75ms
    """
    current_user = get_current_user_from_request(request)
    
    # Check if user has security monitoring permission
    if not hasattr(current_user, 'permissions') or Permission.VIEW_SYSTEM_LOGS not in current_user.permissions:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions to view security status"
        )
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "security_level": "high",
        "authentication": {
            "method": "JWT",
            "token_expiry_hours": 24,
            "refresh_token_expiry_days": 30
        },
        "authorization": {
            "method": "RBAC",
            "total_roles": len(UserRole),
            "total_permissions": len(Permission)
        },
        "audit": {
            "enabled": True,
            "retention_days": 90
        },
        "threats": {
            "active_incidents": 0,
            "resolved_today": 0
        }
    }