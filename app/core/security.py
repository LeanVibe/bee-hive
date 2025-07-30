"""
Security utilities for the application.

Provides enterprise-grade authentication and authorization functionality.
"""

import os
import jwt
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
import logging

logger = logging.getLogger(__name__)

# Security configuration
security_scheme = HTTPBearer(auto_error=False)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Configuration - defer loading until needed to support testing
def get_jwt_config():
    """Get JWT configuration, loading it lazily to support testing."""
    try:
        from .config import settings
        return {
            'secret_key': settings.JWT_SECRET_KEY,
            'algorithm': settings.JWT_ALGORITHM,
            'expire_minutes': settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        }
    except Exception:
        # Fallback to environment variables for backwards compatibility
        jwt_secret = os.getenv("JWT_SECRET_KEY")
        if not jwt_secret:
            # Allow testing without JWT_SECRET_KEY for specific scenarios
            if os.getenv("ENVIRONMENT") == "test":
                jwt_secret = "test-jwt-secret-key-for-testing-purposes-only"
            else:
                raise ValueError("JWT_SECRET_KEY environment variable is required")
        
        return {
            'secret_key': jwt_secret,
            'algorithm': os.getenv("JWT_ALGORITHM", "HS256"),
            'expire_minutes': int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
        }

class SecurityError(Exception):
    """Base security exception."""
    pass

class AuthenticationError(SecurityError):
    """Authentication failed."""
    pass

class AuthorizationError(SecurityError):
    """Authorization failed."""
    pass

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token.
    
    Args:
        data: Token payload data
        expires_delta: Custom expiration time
        
    Returns:
        Encoded JWT token
    """
    config = get_jwt_config()
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=config['expire_minutes'])
    
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    
    encoded_jwt = jwt.encode(to_encode, config['secret_key'], algorithm=config['algorithm'])
    return encoded_jwt

def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify and decode JWT token.
    
    Args:
        token: JWT token to verify
        
    Returns:
        Decoded token payload
        
    Raises:
        AuthenticationError: If token is invalid
    """
    try:
        config = get_jwt_config()
        payload = jwt.decode(token, config['secret_key'], algorithms=[config['algorithm']])
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired")
    except jwt.JWTError:
        raise AuthenticationError("Invalid token")

def hash_password(password: str) -> str:
    """Hash password using bcrypt."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)) -> Optional[Dict[str, Any]]:
    """
    Get current authenticated user from JWT token.
    
    Args:
        credentials: HTTP Authorization credentials
        
    Returns:
        User information if authenticated
        
    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        return None
    
    try:
        token_payload = verify_token(credentials.credentials)
        
        # Validate required fields
        if "sub" not in token_payload or "roles" not in token_payload:
            raise AuthenticationError("Invalid token payload")
        
        return {
            "id": token_payload["sub"],
            "username": token_payload.get("username", token_payload["sub"]),
            "roles": token_payload["roles"],
            "scopes": token_payload.get("scopes", []),
            "human_controller": token_payload.get("human_controller"),
            "expires_at": token_payload.get("exp")
        }
        
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )

async def get_current_active_user(current_user: Optional[Dict[str, Any]] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get current active user (requires authentication).
    
    Args:
        current_user: Current user from token
        
    Returns:
        Active user information
        
    Raises:
        HTTPException: If user not authenticated
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Check if token is expired
    if current_user.get("expires_at"):
        if datetime.utcnow().timestamp() > current_user["expires_at"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"}
            )
    
    return current_user


def verify_analytics_access(user: Dict[str, Any]) -> bool:
    """
    Verify user has access to analytics endpoints.
    
    Args:
        user: User information
        
    Returns:
        True if user has analytics access
    """
    if not user:
        return False
    
    # Check if user has analytics role or admin role
    user_roles = user.get("roles", [])
    return ("analytics_viewer" in user_roles or 
            "admin" in user_roles or 
            "system_admin" in user_roles)

def verify_admin_access(user: Dict[str, Any]) -> bool:
    """
    Verify user has administrative access.
    
    Args:
        user: User information
        
    Returns:
        True if user has admin access
    """
    if not user:
        return False
    
    user_roles = user.get("roles", [])
    return "admin" in user_roles or "system_admin" in user_roles

def require_analytics_access(user: Optional[Dict[str, Any]] = Depends(get_current_active_user)):
    """
    Require analytics access for endpoint.
    
    Args:
        user: User information
        
    Raises:
        HTTPException: If user doesn't have access
    """
    if not user or not verify_analytics_access(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for analytics access"
        )

def require_admin_access(user: Optional[Dict[str, Any]] = Depends(get_current_active_user)):
    """
    Require administrative access for endpoint.
    
    Args:
        user: User information
        
    Raises:
        HTTPException: If user doesn't have admin access
    """
    if not user or not verify_admin_access(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrative privileges required"
        )

def generate_secure_secret(length: int = 64) -> str:
    """
    Generate cryptographically secure secret.
    
    Args:
        length: Length of secret to generate
        
    Returns:
        URL-safe base64 encoded secret
    """
    return secrets.token_urlsafe(length)