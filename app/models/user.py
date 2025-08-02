"""
User model for LeanVibe Agent Hive 2.0

Represents application users with authentication and authorization capabilities.
Supports enterprise-grade user management and access control.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum

from sqlalchemy import Column, String, Text, DateTime, JSON, Boolean, Enum as SQLEnum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from ..core.database import Base
from ..core.database_types import DatabaseAgnosticUUID, StringArray


class UserRole(Enum):
    """User role levels."""
    USER = "user"
    ADMIN = "admin"
    SYSTEM_ADMIN = "system_admin"
    ANALYTICS_VIEWER = "analytics_viewer"
    DEVELOPER = "developer"


class UserStatus(Enum):
    """User account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"


class User(Base):
    """
    Represents an application user with authentication and authorization.
    
    Supports enterprise-grade user management including roles, permissions,
    and audit trails for compliance requirements.
    """
    
    __tablename__ = "users"
    
    # Primary identification
    id = Column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    username = Column(String(255), nullable=False, unique=True, index=True)
    email = Column(String(255), nullable=False, unique=True, index=True)
    
    # Authentication
    password_hash = Column(String(255), nullable=True)  # Nullable for SSO users
    salt = Column(String(255), nullable=True)
    
    # Profile information
    first_name = Column(String(255), nullable=True)
    last_name = Column(String(255), nullable=True)
    display_name = Column(String(255), nullable=True)
    
    # Authorization
    roles = Column(StringArray(), nullable=False, default=list)
    permissions = Column(StringArray(), nullable=True, default=list)
    status = Column(SQLEnum(UserStatus), nullable=False, default=UserStatus.ACTIVE, index=True)
    
    # Profile settings
    preferences = Column(JSON, nullable=True, default=dict)
    settings = Column(JSON, nullable=True, default=dict)
    
    # Enterprise features
    organization_id = Column(String(255), nullable=True, index=True)
    department = Column(String(255), nullable=True)
    job_title = Column(String(255), nullable=True)
    
    # Security and compliance
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    password_changed_at = Column(DateTime(timezone=True), nullable=True)
    failed_login_attempts = Column(String(10), nullable=False, default="0")
    locked_until = Column(DateTime(timezone=True), nullable=True)
    
    # Audit trail
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_by_id = Column(DatabaseAgnosticUUID(), nullable=True)
    updated_by_id = Column(DatabaseAgnosticUUID(), nullable=True)
    
    # Account management
    is_active = Column(Boolean, nullable=False, default=True)
    email_verified = Column(Boolean, nullable=False, default=False)
    terms_accepted_at = Column(DateTime(timezone=True), nullable=True)
    privacy_policy_accepted_at = Column(DateTime(timezone=True), nullable=True)
    
    def __init__(self, **kwargs):
        """Initialize user with proper defaults."""
        # Set default values if not provided
        if 'status' not in kwargs:
            kwargs['status'] = UserStatus.ACTIVE
        if 'roles' not in kwargs:
            kwargs['roles'] = [UserRole.USER.value]
        if 'permissions' not in kwargs:
            kwargs['permissions'] = []
        if 'preferences' not in kwargs:
            kwargs['preferences'] = {}
        if 'settings' not in kwargs:
            kwargs['settings'] = {}
        if 'is_active' not in kwargs:
            kwargs['is_active'] = True
        if 'email_verified' not in kwargs:
            kwargs['email_verified'] = False
        if 'failed_login_attempts' not in kwargs:
            kwargs['failed_login_attempts'] = "0"
        
        super().__init__(**kwargs)
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}', status='{self.status}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary for serialization."""
        return {
            "id": str(self.id),
            "username": self.username,
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "display_name": self.display_name,
            "roles": self.roles,
            "permissions": self.permissions,
            "status": self.status.value,
            "organization_id": self.organization_id,
            "department": self.department,
            "job_title": self.job_title,
            "is_active": self.is_active,
            "email_verified": self.email_verified,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "preferences": self.preferences,
            "settings": self.settings
        }
    
    def has_role(self, role: str) -> bool:
        """Check if user has specific role."""
        if isinstance(role, UserRole):
            role = role.value
        return role in (self.roles or [])
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in (self.permissions or [])
    
    def has_any_role(self, roles: List[str]) -> bool:
        """Check if user has any of the specified roles."""
        user_roles = self.roles or []
        return any(role in user_roles for role in roles)
    
    def is_admin(self) -> bool:
        """Check if user has admin role."""
        return self.has_role(UserRole.ADMIN) or self.has_role(UserRole.SYSTEM_ADMIN)
    
    def is_system_admin(self) -> bool:
        """Check if user has system admin role."""
        return self.has_role(UserRole.SYSTEM_ADMIN)
    
    def can_access_analytics(self) -> bool:
        """Check if user can access analytics."""
        return (self.has_role(UserRole.ANALYTICS_VIEWER) or 
                self.has_role(UserRole.ADMIN) or 
                self.has_role(UserRole.SYSTEM_ADMIN))
    
    def add_role(self, role: str) -> None:
        """Add role to user."""
        if isinstance(role, UserRole):
            role = role.value
        
        if self.roles is None:
            self.roles = []
        
        if role not in self.roles:
            self.roles.append(role)
    
    def remove_role(self, role: str) -> None:
        """Remove role from user."""
        if isinstance(role, UserRole):
            role = role.value
        
        if self.roles and role in self.roles:
            self.roles.remove(role)
    
    def add_permission(self, permission: str) -> None:
        """Add permission to user."""
        if self.permissions is None:
            self.permissions = []
        
        if permission not in self.permissions:
            self.permissions.append(permission)
    
    def remove_permission(self, permission: str) -> None:
        """Remove permission from user."""
        if self.permissions and permission in self.permissions:
            self.permissions.remove(permission)
    
    def is_locked(self) -> bool:
        """Check if user account is locked."""
        if not self.locked_until:
            return False
        return datetime.utcnow() < self.locked_until
    
    def update_last_login(self) -> None:
        """Update last login timestamp."""
        self.last_login_at = datetime.utcnow()
        self.failed_login_attempts = "0"  # Reset failed attempts on successful login
    
    def increment_failed_login(self, max_attempts: int = 5) -> bool:
        """
        Increment failed login attempts and lock account if needed.
        
        Returns:
            True if account should be locked
        """
        try:
            current_attempts = int(self.failed_login_attempts or "0")
        except ValueError:
            current_attempts = 0
        
        current_attempts += 1
        self.failed_login_attempts = str(current_attempts)
        
        if current_attempts >= max_attempts:
            # Lock account for 30 minutes
            from datetime import timedelta
            self.locked_until = datetime.utcnow() + timedelta(minutes=30)
            return True
        
        return False
    
    def unlock_account(self) -> None:
        """Unlock user account and reset failed attempts."""
        self.locked_until = None
        self.failed_login_attempts = "0"
    
    def get_full_name(self) -> str:
        """Get user's full name."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        elif self.display_name:
            return self.display_name
        else:
            return self.username
    
    def can_perform_action(self, action: str, resource: str = None) -> bool:
        """
        Check if user can perform specific action.
        
        Args:
            action: Action to check (e.g., 'read', 'write', 'delete')
            resource: Optional resource type (e.g., 'users', 'agents', 'tasks')
        
        Returns:
            True if user can perform action
        """
        # System admins can do everything
        if self.is_system_admin():
            return True
        
        # Check specific permissions
        if resource:
            permission = f"{action}:{resource}"
            if self.has_permission(permission):
                return True
        
        # Check general permissions
        if self.has_permission(action):
            return True
        
        # Role-based access control
        if self.is_admin():
            # Admins can do most things except system-level operations
            restricted_actions = ['system:shutdown', 'system:configure']
            full_action = f"{action}:{resource}" if resource else action
            return full_action not in restricted_actions
        
        # Regular users have limited permissions
        if action in ['read'] and resource in ['tasks', 'agents', 'sessions']:
            return True
        
        return False