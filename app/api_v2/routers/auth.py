"""
Authentication Router for API v2

Provides JWT-based authentication endpoints with database integration,
user registration, login, logout, and token management for the
consolidated API v2 system.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, status, Request
from pydantic import BaseModel, Field, validator
import structlog

from ...models.user import User, UserRole, UserStatus
from ...services.user_service import get_user_service, UserService
from ...core.database import get_session
from ..middleware import get_current_user_from_request

logger = structlog.get_logger()

router = APIRouter()


# Pydantic models
class UserRegistration(BaseModel):
    """User registration request model."""
    username: str = Field(..., min_length=3, max_length=50, regex=r'^[a-zA-Z0-9_-]+$')
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    password: str = Field(..., min_length=8, max_length=100)
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    
    @validator('password')
    def validate_password(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


class UserLogin(BaseModel):
    """User login request model."""
    email: str
    password: str


class UserResponse(BaseModel):
    """User response model."""
    id: str
    username: str
    email: str
    first_name: Optional[str]
    last_name: Optional[str]
    display_name: Optional[str]
    roles: list[str]
    permissions: list[str] = []
    status: str
    is_active: bool
    email_verified: bool
    last_login_at: Optional[datetime]
    created_at: datetime
    
    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int
    user: UserResponse


class RefreshRequest(BaseModel):
    """Refresh token request model."""
    refresh_token: str


class PasswordChangeRequest(BaseModel):
    """Password change request model."""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)
    
    @validator('new_password')
    def validate_new_password(cls, v):
        """Validate new password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


# Authentication endpoints
@router.post("/register", response_model=Dict[str, Any], tags=["Authentication"])
async def register_user(registration: UserRegistration) -> Dict[str, Any]:
    """
    Register new user account.
    
    Creates a new user account with email verification required.
    Returns success status and user information (without sensitive data).
    """
    user_service = get_user_service()
    
    async with get_session() as db:
        try:
            user = await user_service.create_user(
                db=db,
                username=registration.username,
                email=registration.email,
                password=registration.password,
                first_name=registration.first_name,
                last_name=registration.last_name
            )
            
            logger.info("User registered successfully", 
                       user_id=str(user.id), 
                       username=registration.username,
                       email=registration.email)
            
            return {
                "success": True,
                "message": "User registered successfully",
                "user": {
                    "id": str(user.id),
                    "username": user.username,
                    "email": user.email,
                    "email_verified": user.email_verified
                },
                "next_steps": [
                    "Please verify your email address",
                    "Check your inbox for verification link"
                ]
            }
            
        except ValueError as e:
            logger.warning("User registration failed", 
                          username=registration.username,
                          email=registration.email,
                          error=str(e))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error("Unexpected error during registration", 
                        username=registration.username,
                        error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Registration failed due to server error"
            )


@router.post("/login", response_model=TokenResponse, tags=["Authentication"])
async def login_user(login_data: UserLogin) -> TokenResponse:
    """
    Authenticate user and return access tokens.
    
    Validates user credentials and returns JWT access and refresh tokens
    along with user information for successful authentication.
    """
    user_service = get_user_service()
    
    async with get_session() as db:
        user = await user_service.authenticate_user(
            db, login_data.email, login_data.password
        )
        
        if not user:
            logger.warning("Login attempt failed", email=login_data.email)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Check if user account is active
        if not user.is_active or user.status != UserStatus.ACTIVE:
            logger.warning("Login attempt for inactive account", 
                          user_id=str(user.id), email=login_data.email)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is not active"
            )
        
        # Generate tokens
        access_token = user_service.create_access_token(user)
        refresh_token = user_service.create_refresh_token(user)
        
        logger.info("User login successful", 
                   user_id=str(user.id), 
                   email=login_data.email)
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=24 * 3600,  # 24 hours
            user=UserResponse(
                id=str(user.id),
                username=user.username,
                email=user.email,
                first_name=user.first_name,
                last_name=user.last_name,
                display_name=user.display_name,
                roles=user.roles or [],
                permissions=user.permissions or [],
                status=user.status.value,
                is_active=user.is_active,
                email_verified=user.email_verified,
                last_login_at=user.last_login_at,
                created_at=user.created_at
            )
        )


@router.post("/refresh", response_model=Dict[str, Any], tags=["Authentication"])
async def refresh_token(refresh_request: RefreshRequest) -> Dict[str, Any]:
    """
    Refresh access token using refresh token.
    
    Validates refresh token and returns new access token if valid.
    """
    user_service = get_user_service()
    
    async with get_session() as db:
        result = await user_service.refresh_access_token(
            db, refresh_request.refresh_token
        )
        
        if not result:
            logger.warning("Token refresh failed - invalid refresh token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token"
            )
        
        logger.debug("Token refreshed successfully")
        
        return {
            "success": True,
            **result
        }


@router.get("/me", response_model=UserResponse, tags=["Authentication"])
async def get_current_user(request: Request) -> UserResponse:
    """
    Get current authenticated user information.
    
    Returns detailed user profile information for the authenticated user.
    """
    user = get_current_user_from_request(request)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    return UserResponse(
        id=str(user.id),
        username=user.username,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
        display_name=user.display_name,
        roles=user.roles or [],
        permissions=user.permissions or [],
        status=user.status.value,
        is_active=user.is_active,
        email_verified=user.email_verified,
        last_login_at=user.last_login_at,
        created_at=user.created_at
    )


@router.post("/logout", response_model=Dict[str, Any], tags=["Authentication"])
async def logout_user(request: Request) -> Dict[str, Any]:
    """
    Logout current user.
    
    With stateless JWT tokens, this endpoint primarily serves as a client-side
    logout confirmation. Clients should delete tokens locally.
    """
    user = get_current_user_from_request(request)
    
    if user:
        logger.info("User logged out", user_id=str(user.id), email=user.email)
    
    return {
        "success": True,
        "message": "Logged out successfully",
        "instructions": [
            "Clear tokens from client storage",
            "Redirect to login page if needed"
        ]
    }


@router.post("/change-password", response_model=Dict[str, Any], tags=["Authentication"])
async def change_password(
    password_request: PasswordChangeRequest,
    request: Request
) -> Dict[str, Any]:
    """
    Change user password.
    
    Validates current password and updates to new password if valid.
    """
    user = get_current_user_from_request(request)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    user_service = get_user_service()
    
    # Verify current password
    if not user_service.verify_password(password_request.current_password, user.password_hash):
        logger.warning("Password change failed - invalid current password", 
                      user_id=str(user.id))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    async with get_session() as db:
        success = await user_service.change_password(
            db, str(user.id), password_request.new_password
        )
        
        if not success:
            logger.error("Password change failed - database error", 
                        user_id=str(user.id))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update password"
            )
    
    logger.info("Password changed successfully", user_id=str(user.id))
    
    return {
        "success": True,
        "message": "Password changed successfully",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/verify-email/{user_id}", response_model=Dict[str, Any], tags=["Authentication"])
async def verify_email(user_id: str) -> Dict[str, Any]:
    """
    Verify user email address.
    
    In a production system, this would validate an email verification token.
    For now, it directly marks the email as verified.
    """
    user_service = get_user_service()
    
    async with get_session() as db:
        success = await user_service.verify_email(db, user_id)
        
        if not success:
            logger.error("Email verification failed", user_id=user_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found or verification failed"
            )
    
    logger.info("Email verified successfully", user_id=user_id)
    
    return {
        "success": True,
        "message": "Email verified successfully",
        "timestamp": datetime.utcnow().isoformat()
    }


# Health check for authentication service
@router.get("/health", response_model=Dict[str, Any], tags=["Authentication"])
async def auth_health_check() -> Dict[str, Any]:
    """
    Authentication service health check.
    
    Verifies that the authentication service and database connectivity
    are functioning properly.
    """
    try:
        # Test database connectivity
        async with get_session() as db:
            # Simple query to test connection
            await db.execute("SELECT 1")
        
        # Test user service initialization
        user_service = get_user_service()
        
        return {
            "status": "healthy",
            "service": "authentication",
            "database": "connected",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "2.0.0"
        }
        
    except Exception as e:
        logger.error("Authentication service health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "service": "authentication",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


__all__ = ["router"]