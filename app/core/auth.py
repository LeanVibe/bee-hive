# ruff: noqa
"""
Authentication and Authorization System for LeanVibe Agent Hive 2.0

Implements JWT-based authentication, RBAC authorization, and enterprise SSO
integration for secure access to autonomous development platform.

CRITICAL COMPONENT: Provides security layer for all enterprise operations.
"""

import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union

import structlog
import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from enum import Enum

from .database import get_session
from .auth_metrics import inc as inc_auth_metric

logger = structlog.get_logger()

# Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_HOURS = 24
JWT_REFRESH_TOKEN_EXPIRE_DAYS = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer token extraction
security = HTTPBearer(auto_error=False)


# Enums
class UserRole(Enum):
    """User roles for RBAC."""
    SUPER_ADMIN = "super_admin"
    ENTERPRISE_ADMIN = "enterprise_admin"
    PILOT_MANAGER = "pilot_manager"
    SUCCESS_MANAGER = "success_manager"
    DEVELOPER = "developer"
    VIEWER = "viewer"


class Permission(Enum):
    """System permissions."""
    # Pilot management
    CREATE_PILOT = "create_pilot"
    VIEW_PILOT = "view_pilot"
    UPDATE_PILOT = "update_pilot"
    DELETE_PILOT = "delete_pilot"
    
    # ROI and analytics
    VIEW_ROI_METRICS = "view_roi_metrics"
    CREATE_ROI_METRICS = "create_roi_metrics"
    VIEW_ANALYTICS = "view_analytics"
    
    # Executive engagement
    VIEW_EXECUTIVE_ENGAGEMENT = "view_executive_engagement"
    CREATE_EXECUTIVE_ENGAGEMENT = "create_executive_engagement"
    UPDATE_EXECUTIVE_ENGAGEMENT = "update_executive_engagement"
    
    # Development tasks
    CREATE_DEVELOPMENT_TASK = "create_development_task"
    VIEW_DEVELOPMENT_TASK = "view_development_task"
    EXECUTE_DEVELOPMENT_TASK = "execute_development_task"
    
    # System administration
    MANAGE_USERS = "manage_users"
    VIEW_SYSTEM_LOGS = "view_system_logs"
    CONFIGURE_SYSTEM = "configure_system"


# Role-Permission Mapping
ROLE_PERMISSIONS = {
    UserRole.SUPER_ADMIN: list(Permission),  # All permissions
    UserRole.ENTERPRISE_ADMIN: [
        Permission.CREATE_PILOT, Permission.VIEW_PILOT, Permission.UPDATE_PILOT,
        Permission.VIEW_ROI_METRICS, Permission.CREATE_ROI_METRICS, Permission.VIEW_ANALYTICS,
        Permission.VIEW_EXECUTIVE_ENGAGEMENT, Permission.CREATE_EXECUTIVE_ENGAGEMENT,
        Permission.UPDATE_EXECUTIVE_ENGAGEMENT,
        Permission.CREATE_DEVELOPMENT_TASK, Permission.VIEW_DEVELOPMENT_TASK,
        Permission.EXECUTE_DEVELOPMENT_TASK, Permission.MANAGE_USERS
    ],
    UserRole.PILOT_MANAGER: [
        Permission.CREATE_PILOT, Permission.VIEW_PILOT, Permission.UPDATE_PILOT,
        Permission.VIEW_ROI_METRICS, Permission.CREATE_ROI_METRICS,
        Permission.CREATE_DEVELOPMENT_TASK, Permission.VIEW_DEVELOPMENT_TASK,
        Permission.EXECUTE_DEVELOPMENT_TASK
    ],
    UserRole.SUCCESS_MANAGER: [
        Permission.VIEW_PILOT, Permission.UPDATE_PILOT,
        Permission.VIEW_ROI_METRICS, Permission.CREATE_ROI_METRICS, Permission.VIEW_ANALYTICS,
        Permission.VIEW_EXECUTIVE_ENGAGEMENT, Permission.CREATE_EXECUTIVE_ENGAGEMENT,
        Permission.UPDATE_EXECUTIVE_ENGAGEMENT
    ],
    UserRole.DEVELOPER: [
        Permission.VIEW_PILOT, Permission.CREATE_DEVELOPMENT_TASK,
        Permission.VIEW_DEVELOPMENT_TASK, Permission.EXECUTE_DEVELOPMENT_TASK
    ],
    UserRole.VIEWER: [
        Permission.VIEW_PILOT, Permission.VIEW_ROI_METRICS, Permission.VIEW_ANALYTICS,
        Permission.VIEW_EXECUTIVE_ENGAGEMENT, Permission.VIEW_DEVELOPMENT_TASK
    ]
}


# Pydantic Models
class UserCreate(BaseModel):
    """User creation model."""
    email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=1, max_length=255)
    role: UserRole = UserRole.VIEWER
    company_name: Optional[str] = None
    pilot_ids: List[str] = []  # Pilots this user can access
    
    @validator('password')
    def validate_password(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


class UserLogin(BaseModel):
    """User login model."""
    email: str
    password: str


class UserResponse(BaseModel):
    """User response model."""
    id: str
    email: str
    full_name: str
    role: UserRole
    company_name: Optional[str]
    pilot_ids: List[str]
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int
    user: UserResponse


class TokenData(BaseModel):
    """Token data model."""
    user_id: str
    email: str
    role: UserRole
    pilot_ids: List[str]
    permissions: List[Permission]
    exp: datetime


class RefreshRequest(BaseModel):
    """Request body for refresh token endpoint."""
    refresh_token: str


# Database Models (Simple in-memory for now, should be moved to database_models.py)
class User:
    """User model for authentication."""
    
    def __init__(self, id: str, email: str, hashed_password: str, full_name: str,
                 role: UserRole, company_name: Optional[str] = None, 
                 pilot_ids: List[str] = None, is_active: bool = True):
        self.id = id
        self.email = email
        self.hashed_password = hashed_password
        self.full_name = full_name
        self.role = role
        self.company_name = company_name
        self.pilot_ids = pilot_ids or []
        self.is_active = is_active
        self.created_at = datetime.utcnow()
        self.last_login: Optional[datetime] = None


# In-memory user store (should be replaced with database)
_users_store: Dict[str, User] = {}


class AuthenticationService:
    """
    Comprehensive authentication and authorization service.
    
    Provides JWT token management, password hashing, role-based access control,
    and enterprise SSO integration foundation.
    """
    
    def __init__(self):
        self.pwd_context = pwd_context
        self.secret_key = JWT_SECRET_KEY
        self.algorithm = JWT_ALGORITHM
        
        # Create default admin user if none exists
        self._ensure_default_admin()
    
    def _ensure_default_admin(self):
        """Ensure default admin user exists."""
        admin_email = os.getenv("DEFAULT_ADMIN_EMAIL", "admin@leanvibe.com")
        admin_password = os.getenv("DEFAULT_ADMIN_PASSWORD", "AdminPassword123!")
        
        if not any(user.email == admin_email for user in _users_store.values()):
            admin_user = User(
                id=str(uuid.uuid4()),
                email=admin_email,
                hashed_password=self.hash_password(admin_password),
                full_name="System Administrator",
                role=UserRole.SUPER_ADMIN,
                company_name="LeanVibe",
                is_active=True
            )
            _users_store[admin_user.id] = admin_user
            
            logger.info("Default admin user created", email=admin_email)
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        for user in _users_store.values():
            if user.email == email:
                return user
        return None
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return _users_store.get(user_id)
    
    def create_user(self, user_data: UserCreate) -> User:
        """Create new user."""
        # Check if user already exists
        if self.get_user_by_email(user_data.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        
        # Create user
        user = User(
            id=str(uuid.uuid4()),
            email=user_data.email,
            hashed_password=self.hash_password(user_data.password),
            full_name=user_data.full_name,
            role=user_data.role,
            company_name=user_data.company_name,
            pilot_ids=user_data.pilot_ids
        )
        
        _users_store[user.id] = user
        
        logger.info("User created", user_id=user.id, email=user.email, role=user.role.value)
        return user
    
    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user credentials."""
        user = self.get_user_by_email(email)
        if not user or not user.is_active:
            return None
        
        if not self.verify_password(password, user.hashed_password):
            return None
        
        # Update last login
        user.last_login = datetime.utcnow()
        return user
    
    def create_access_token(self, user: User) -> str:
        """Create JWT access token."""
        expire = datetime.utcnow() + timedelta(hours=JWT_ACCESS_TOKEN_EXPIRE_HOURS)
        
        payload = {
            "sub": user.id,
            "email": user.email,
            "role": user.role.value,
            "pilot_ids": user.pilot_ids,
            "permissions": [p.value for p in ROLE_PERMISSIONS.get(user.role, [])],
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user: User) -> str:
        """Create JWT refresh token."""
        expire = datetime.utcnow() + timedelta(days=JWT_REFRESH_TOKEN_EXPIRE_DAYS)
        
        payload = {
            "sub": user.id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            user_id = payload.get("sub")
            if not user_id:
                return None
            
            user = self.get_user_by_id(user_id)
            if not user or not user.is_active:
                return None
            
            return TokenData(
                user_id=user_id,
                email=payload.get("email", ""),
                role=UserRole(payload.get("role", UserRole.VIEWER.value)),
                pilot_ids=payload.get("pilot_ids", []),
                permissions=[Permission(p) for p in payload.get("permissions", [])],
                exp=datetime.fromtimestamp(payload.get("exp", 0))
            )
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning("Invalid token", error=str(e))
            return None
    
    def user_has_permission(self, user: User, required_permission: Permission) -> bool:
        """Check if user has required permission."""
        user_permissions = ROLE_PERMISSIONS.get(user.role, [])
        return required_permission in user_permissions
    
    def user_can_access_pilot(self, user: User, pilot_id: str) -> bool:
        """Check if user can access specific pilot."""
        # Super admins can access all pilots
        if user.role == UserRole.SUPER_ADMIN:
            return True
        
        # Check if pilot is in user's allowed pilot list
        return pilot_id in user.pilot_ids


# Global authentication service instance
_auth_service: Optional[AuthenticationService] = None


def get_auth_service() -> AuthenticationService:
    """Get or create authentication service instance."""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthenticationService()
    return _auth_service


# FastAPI Dependencies
async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> User:
    """Get current authenticated user from JWT token."""
    if credentials is None or not getattr(credentials, "credentials", None):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    auth_service = get_auth_service()
    token_data = auth_service.verify_token(credentials.credentials)
    
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    user = auth_service.get_user_by_id(token_data.user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return user


def require_permission(required_permission: Permission):
    """Dependency factory for requiring specific permissions."""
    
    def permission_checker(current_user: User = Depends(get_current_user)) -> User:
        auth_service = get_auth_service()
        
        if not auth_service.user_has_permission(current_user, required_permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {required_permission.value}"
            )
        
        return current_user
    
    return permission_checker


def require_pilot_access(pilot_id: str):
    """Dependency factory for requiring access to specific pilot."""
    
    def pilot_access_checker(current_user: User = Depends(get_current_user)) -> User:
        auth_service = get_auth_service()
        
        if not auth_service.user_can_access_pilot(current_user, pilot_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied to pilot: {pilot_id}"
            )
        
        return current_user
    
    return pilot_access_checker


# Authentication API endpoints
from fastapi import APIRouter

auth_router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


@auth_router.post("/register", response_model=Dict[str, Any])
async def register_user(user_data: UserCreate) -> Dict[str, Any]:
    """Register new user."""
    
    auth_service = get_auth_service()
    
    try:
        user = auth_service.create_user(user_data)
        
        return {
            "success": True,
            "message": "User registered successfully",
            "user": {
                "id": user.id,
                "email": user.email,
                "full_name": user.full_name,
                "role": user.role.value
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("User registration failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@auth_router.post("/login", response_model=TokenResponse)
async def login_user(login_data: UserLogin) -> TokenResponse:
    """Authenticate user and return JWT tokens."""
    
    auth_service = get_auth_service()
    
    user = auth_service.authenticate_user(login_data.email, login_data.password)
    if not user:
        try:
            inc_auth_metric("auth_failure_total_rest")
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Create tokens
    access_token = auth_service.create_access_token(user)
    refresh_token = auth_service.create_refresh_token(user)
    
    logger.info("User logged in", user_id=user.id, email=user.email)
    try:
        inc_auth_metric("auth_success_total_rest")
    except Exception:
        pass
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=JWT_ACCESS_TOKEN_EXPIRE_HOURS * 3600,
        user=UserResponse(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            company_name=user.company_name,
            pilot_ids=user.pilot_ids,
            is_active=user.is_active,
            created_at=user.created_at,
            last_login=user.last_login
        )
    )


@auth_router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
) -> UserResponse:
    """Get current user information."""
    
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        company_name=current_user.company_name,
        pilot_ids=current_user.pilot_ids,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
        last_login=current_user.last_login
    )


@auth_router.post("/refresh", response_model=Dict[str, Any])
async def refresh_access_token(payload: RefreshRequest) -> Dict[str, Any]:
    """Refresh access token using refresh token."""
    
    auth_service = get_auth_service()
    
    try:
        decoded = jwt.decode(payload.refresh_token, auth_service.secret_key, algorithms=[auth_service.algorithm])
        
        if decoded.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        user_id = decoded.get("sub")
        user = auth_service.get_user_by_id(user_id)
        
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Create new access token
        new_access_token = auth_service.create_access_token(user)
        
        return {
            "success": True,
            "access_token": new_access_token,
            "token_type": "Bearer",
            "expires_in": JWT_ACCESS_TOKEN_EXPIRE_HOURS * 3600
        }
        
    except jwt.ExpiredSignatureError:
        try:
            inc_auth_metric("auth_failure_total_rest")
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token expired"
        )
    except jwt.InvalidTokenError:
        try:
            inc_auth_metric("auth_failure_total_rest")
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )


@auth_router.post("/logout", response_model=Dict[str, Any])
async def logout_user(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Logout current user. With stateless JWT we just acknowledge."""
    logger.info("User logged out", user_id=current_user.id, email=current_user.email)
    return {"success": True}


class SecurityAuditEntry(BaseModel):
    """Security audit log entry shape from PWA."""
    timestamp: datetime
    event: str
    userId: Optional[str] = None
    userAgent: Optional[str] = None
    ipAddress: Optional[str] = None
    success: bool
    details: Optional[Any] = None


@auth_router.post("/audit", response_model=Dict[str, Any])
async def audit_security_event(entry: SecurityAuditEntry) -> Dict[str, Any]:
    """Accept security audit log entries from clients (best-effort)."""
    try:
        logger.info(
            "security_audit_log",
            event=entry.event,
            success=entry.success,
            user_id=entry.userId,
            user_agent=entry.userAgent,
            ip_address=entry.ipAddress,
            details=entry.details,
        )
        return {"success": True}
    except Exception as e:
        logger.error("Failed to record security audit log", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to record audit log")


# Standalone functions for middleware compatibility
def decode_jwt_token(token: str) -> Optional[TokenData]:
    """Decode JWT token - compatibility wrapper for middleware."""
    auth_service = get_auth_service()
    return auth_service.verify_token(token)


def get_user_by_id(user_id: str) -> Optional[User]:
    """Get user by ID - compatibility wrapper for middleware."""
    auth_service = get_auth_service()
    return auth_service.get_user_by_id(user_id)


# Export all authentication components
__all__ = [
    "AuthenticationService", "get_auth_service", "get_current_user",
    "require_permission", "require_pilot_access", "auth_router",
    "UserRole", "Permission", "User", "UserCreate", "UserLogin",
    "UserResponse", "TokenResponse", "decode_jwt_token", "get_user_by_id"
]