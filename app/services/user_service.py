"""
User Service for LeanVibe Agent Hive 2.0

Database-integrated user management service that supports authentication,
authorization, and user lifecycle management with PostgreSQL persistence.
"""

import uuid
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from passlib.context import CryptContext
import jwt
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from sqlalchemy.exc import IntegrityError

from ..models.user import User, UserRole, UserStatus
from ..core.database import get_session
from ..core.auth_metrics import inc as inc_auth_metric

logger = structlog.get_logger()

# Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_HOURS = 24
JWT_REFRESH_TOKEN_EXPIRE_DAYS = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserService:
    """
    Database-integrated user service for authentication and authorization.
    
    Provides comprehensive user management with PostgreSQL persistence,
    JWT token management, and enterprise-grade security features.
    """
    
    def __init__(self):
        self.pwd_context = pwd_context
        self.secret_key = JWT_SECRET_KEY
        self.algorithm = JWT_ALGORITHM
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    async def create_user(
        self,
        db: AsyncSession,
        username: str,
        email: str,
        password: str,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        roles: Optional[List[str]] = None
    ) -> User:
        """
        Create new user in database.
        
        Args:
            db: Database session
            username: Unique username
            email: User email address
            password: Plain text password (will be hashed)
            first_name: Optional first name
            last_name: Optional last name
            roles: Optional list of roles (defaults to ['user'])
        
        Returns:
            Created User instance
        
        Raises:
            ValueError: If user already exists
        """
        # Check if user already exists
        existing_user = await self.get_user_by_email(db, email)
        if existing_user:
            raise ValueError(f"User with email {email} already exists")
        
        existing_username = await self.get_user_by_username(db, username)
        if existing_username:
            raise ValueError(f"User with username {username} already exists")
        
        # Set default roles
        if not roles:
            roles = [UserRole.USER.value]
        
        # Create user
        user = User(
            username=username,
            email=email,
            password_hash=self.hash_password(password),
            first_name=first_name,
            last_name=last_name,
            roles=roles,
            status=UserStatus.ACTIVE,
            email_verified=False,
            is_active=True
        )
        
        try:
            db.add(user)
            await db.commit()
            await db.refresh(user)
            
            logger.info("User created successfully", 
                       user_id=str(user.id), 
                       username=username, 
                       email=email)
            
            return user
            
        except IntegrityError as e:
            await db.rollback()
            logger.error("Failed to create user due to integrity error", error=str(e))
            raise ValueError("Failed to create user: database integrity error")
    
    async def get_user_by_id(self, db: AsyncSession, user_id: str) -> Optional[User]:
        """Get user by ID."""
        try:
            uuid_id = uuid.UUID(user_id) if isinstance(user_id, str) else user_id
            result = await db.execute(select(User).where(User.id == uuid_id))
            return result.scalar_one_or_none()
        except (ValueError, TypeError):
            return None
    
    async def get_user_by_email(self, db: AsyncSession, email: str) -> Optional[User]:
        """Get user by email."""
        result = await db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()
    
    async def get_user_by_username(self, db: AsyncSession, username: str) -> Optional[User]:
        """Get user by username."""
        result = await db.execute(select(User).where(User.username == username))
        return result.scalar_one_or_none()
    
    async def authenticate_user(self, db: AsyncSession, email: str, password: str) -> Optional[User]:
        """
        Authenticate user with email and password.
        
        Args:
            db: Database session
            email: User email
            password: Plain text password
        
        Returns:
            User instance if authentication successful, None otherwise
        """
        user = await self.get_user_by_email(db, email)
        
        if not user or not user.is_active:
            try:
                inc_auth_metric("auth_failure_total")
            except Exception:
                pass
            return None
        
        # Check if account is locked
        if user.is_locked():
            logger.warning("Authentication failed: account locked", 
                          user_id=str(user.id), email=email)
            try:
                inc_auth_metric("auth_failure_total")
            except Exception:
                pass
            return None
        
        # Verify password
        if not self.verify_password(password, user.password_hash):
            # Increment failed login attempts
            should_lock = user.increment_failed_login()
            if should_lock:
                logger.warning("Account locked due to too many failed attempts", 
                              user_id=str(user.id), email=email)
            
            await db.commit()
            try:
                inc_auth_metric("auth_failure_total")
            except Exception:
                pass
            return None
        
        # Authentication successful
        user.update_last_login()
        await db.commit()
        
        logger.info("User authenticated successfully", 
                   user_id=str(user.id), email=email)
        try:
            inc_auth_metric("auth_success_total")
        except Exception:
            pass
        
        return user
    
    def create_access_token(self, user: User) -> str:
        """Create JWT access token."""
        expire = datetime.utcnow() + timedelta(hours=JWT_ACCESS_TOKEN_EXPIRE_HOURS)
        
        payload = {
            "sub": str(user.id),
            "email": user.email,
            "username": user.username,
            "roles": user.roles or [],
            "permissions": user.permissions or [],
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user: User) -> str:
        """Create JWT refresh token."""
        expire = datetime.utcnow() + timedelta(days=JWT_REFRESH_TOKEN_EXPIRE_DAYS)
        
        payload = {
            "sub": str(user.id),
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.debug("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.debug("Invalid token", error=str(e))
            return None
    
    async def refresh_access_token(self, db: AsyncSession, refresh_token: str) -> Optional[Dict[str, Any]]:
        """
        Refresh access token using refresh token.
        
        Returns:
            Dict with new access token if successful, None otherwise
        """
        payload = self.verify_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            return None
        
        user_id = payload.get("sub")
        user = await self.get_user_by_id(db, user_id)
        
        if not user or not user.is_active:
            return None
        
        new_access_token = self.create_access_token(user)
        
        return {
            "access_token": new_access_token,
            "token_type": "Bearer",
            "expires_in": JWT_ACCESS_TOKEN_EXPIRE_HOURS * 3600
        }
    
    async def update_user(
        self,
        db: AsyncSession,
        user_id: str,
        updates: Dict[str, Any]
    ) -> Optional[User]:
        """Update user information."""
        user = await self.get_user_by_id(db, user_id)
        if not user:
            return None
        
        for field, value in updates.items():
            if hasattr(user, field):
                setattr(user, field, value)
        
        user.updated_at = datetime.utcnow()
        
        try:
            await db.commit()
            await db.refresh(user)
            
            logger.info("User updated successfully", 
                       user_id=user_id, 
                       updated_fields=list(updates.keys()))
            
            return user
            
        except IntegrityError as e:
            await db.rollback()
            logger.error("Failed to update user", user_id=user_id, error=str(e))
            return None
    
    async def verify_email(self, db: AsyncSession, user_id: str) -> bool:
        """Mark user email as verified."""
        result = await self.update_user(db, user_id, {"email_verified": True})
        return result is not None
    
    async def change_password(self, db: AsyncSession, user_id: str, new_password: str) -> bool:
        """Change user password."""
        hashed_password = self.hash_password(new_password)
        result = await self.update_user(db, user_id, {
            "password_hash": hashed_password,
            "password_changed_at": datetime.utcnow()
        })
        return result is not None
    
    async def deactivate_user(self, db: AsyncSession, user_id: str) -> bool:
        """Deactivate user account."""
        result = await self.update_user(db, user_id, {
            "is_active": False,
            "status": UserStatus.INACTIVE
        })
        return result is not None
    
    async def unlock_user_account(self, db: AsyncSession, user_id: str) -> bool:
        """Unlock user account and reset failed attempts."""
        user = await self.get_user_by_id(db, user_id)
        if not user:
            return False
        
        user.unlock_account()
        
        try:
            await db.commit()
            logger.info("User account unlocked", user_id=user_id)
            return True
        except Exception as e:
            await db.rollback()
            logger.error("Failed to unlock user account", user_id=user_id, error=str(e))
            return False
    
    async def list_users(
        self,
        db: AsyncSession,
        offset: int = 0,
        limit: int = 100,
        status_filter: Optional[UserStatus] = None
    ) -> List[User]:
        """List users with pagination and filtering."""
        query = select(User)
        
        if status_filter:
            query = query.where(User.status == status_filter)
        
        query = query.offset(offset).limit(limit)
        result = await db.execute(query)
        return result.scalars().all()
    
    def user_has_role(self, user: User, role: str) -> bool:
        """Check if user has specific role."""
        if isinstance(role, UserRole):
            role = role.value
        return role in (user.roles or [])
    
    def user_has_permission(self, user: User, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in (user.permissions or [])
    
    def user_can_access_resource(self, user: User, resource: str, action: str) -> bool:
        """Check if user can perform action on resource."""
        # System admins can do everything
        if self.user_has_role(user, UserRole.SYSTEM_ADMIN):
            return True
        
        # Check specific permission
        permission = f"{action}:{resource}"
        if self.user_has_permission(user, permission):
            return True
        
        # Role-based access control
        if self.user_has_role(user, UserRole.ADMIN):
            # Admins can do most things except system-level operations
            restricted_actions = ["system:shutdown", "system:configure"]
            return permission not in restricted_actions
        
        # Regular users have limited permissions
        if action in ["read"] and resource in ["tasks", "agents", "sessions"]:
            return True
        
        return False


# Global service instance
_user_service: Optional[UserService] = None


def get_user_service() -> UserService:
    """Get or create user service instance."""
    global _user_service
    if _user_service is None:
        _user_service = UserService()
    return _user_service


async def create_default_admin():
    """Create default admin user if none exists."""
    admin_email = os.getenv("DEFAULT_ADMIN_EMAIL", "admin@leanvibe.com")
    admin_password = os.getenv("DEFAULT_ADMIN_PASSWORD", "AdminPassword123!")
    
    user_service = get_user_service()
    
    async with get_session() as db:
        # Check if admin user already exists
        existing_admin = await user_service.get_user_by_email(db, admin_email)
        if existing_admin:
            logger.info("Default admin user already exists", email=admin_email)
            return existing_admin
        
        try:
            admin_user = await user_service.create_user(
                db=db,
                username="admin",
                email=admin_email,
                password=admin_password,
                first_name="System",
                last_name="Administrator",
                roles=[UserRole.SYSTEM_ADMIN.value, UserRole.ADMIN.value]
            )
            
            logger.info("Default admin user created successfully", 
                       user_id=str(admin_user.id), 
                       email=admin_email)
            
            return admin_user
            
        except ValueError as e:
            logger.error("Failed to create default admin user", error=str(e))
            return None


__all__ = [
    "UserService",
    "get_user_service", 
    "create_default_admin",
    "JWT_ACCESS_TOKEN_EXPIRE_HOURS",
    "JWT_REFRESH_TOKEN_EXPIRE_DAYS"
]