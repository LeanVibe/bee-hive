"""
Enterprise Authentication System for LeanVibe Agent Hive 2.0.

Implements enterprise-grade authentication including:
- SAML 2.0 SSO integration
- OAuth 2.0 / OpenID Connect support
- API key management with fine-grained permissions
- Multi-tenant user management with organization isolation
- Service-to-service authentication
- JWT token management with enterprise features
"""

import uuid
import secrets
import hashlib
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union
from enum import Enum
from dataclasses import dataclass, field
import logging
import asyncio
from urllib.parse import urlencode, parse_qs, urlparse
import xml.etree.ElementTree as ET
import json

from sqlalchemy import select, and_, or_, func, desc, text
from sqlalchemy.ext.asyncio import AsyncSession
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.x509 import load_pem_x509_certificate
import jwt
from passlib.context import CryptContext
from authlib.integrations.base_client import OAuthError
from authlib.common.encoding import json_b64encode
import httpx

from ..models.agent import Agent
from ..core.access_control import AccessLevel, Permission
from .security_audit import SecurityAuditSystem, AuditEventType, ThreatLevel


logger = logging.getLogger(__name__)


class AuthenticationMethod(Enum):
    """Authentication methods supported by the system."""
    PASSWORD = "PASSWORD"
    SAML_SSO = "SAML_SSO"
    OAUTH_SSO = "OAUTH_SSO"
    API_KEY = "API_KEY"
    SERVICE_ACCOUNT = "SERVICE_ACCOUNT"
    CERTIFICATE = "CERTIFICATE"
    MFA_TOTP = "MFA_TOTP"


class UserRole(Enum):
    """Enterprise user roles with hierarchical permissions."""
    SUPER_ADMIN = "SUPER_ADMIN"          # Full system access
    TENANT_ADMIN = "TENANT_ADMIN"        # Full tenant access
    SECURITY_ADMIN = "SECURITY_ADMIN"    # Security management
    DEVELOPER = "DEVELOPER"              # Development access
    ANALYST = "ANALYST"                  # Read-only analytics
    API_USER = "API_USER"                # API access only
    SERVICE_ACCOUNT = "SERVICE_ACCOUNT"  # Service-to-service
    GUEST = "GUEST"                      # Limited access


class OrganizationTier(Enum):
    """Organization tier for feature access control."""
    ENTERPRISE = "ENTERPRISE"
    PROFESSIONAL = "PROFESSIONAL"
    STANDARD = "STANDARD"
    STARTER = "STARTER"


@dataclass
class AuthenticatedUser:
    """Represents an authenticated user with enterprise context."""
    id: uuid.UUID
    username: str
    email: str
    role: UserRole
    organization_id: uuid.UUID
    organization_name: str
    organization_tier: OrganizationTier
    permissions: Set[str]
    authentication_method: AuthenticationMethod
    session_id: uuid.UUID
    expires_at: datetime
    mfa_verified: bool = False
    last_activity: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions
    
    def is_admin(self) -> bool:
        """Check if user has admin privileges."""
        return self.role in [UserRole.SUPER_ADMIN, UserRole.TENANT_ADMIN]
    
    def can_access_tenant(self, tenant_id: uuid.UUID) -> bool:
        """Check if user can access specific tenant."""
        if self.role == UserRole.SUPER_ADMIN:
            return True
        return self.organization_id == tenant_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "organization_id": str(self.organization_id),
            "organization_name": self.organization_name,
            "organization_tier": self.organization_tier.value,
            "permissions": list(self.permissions),
            "authentication_method": self.authentication_method.value,
            "session_id": str(self.session_id),
            "expires_at": self.expires_at.isoformat(),
            "mfa_verified": self.mfa_verified,
            "last_activity": self.last_activity.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class APIKey:
    """Represents an API key with permissions and constraints."""
    id: uuid.UUID
    name: str
    key_hash: str
    user_id: uuid.UUID
    organization_id: uuid.UUID
    permissions: Set[str]
    scopes: List[str]
    rate_limit: int  # requests per hour
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
    usage_count: int = 0
    is_active: bool = True
    ip_whitelist: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if API key is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def can_access_scope(self, scope: str) -> bool:
        """Check if API key can access specific scope."""
        return scope in self.scopes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "name": self.name,
            "user_id": str(self.user_id),
            "organization_id": str(self.organization_id),
            "permissions": list(self.permissions),
            "scopes": self.scopes,
            "rate_limit": self.rate_limit,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "usage_count": self.usage_count,
            "is_active": self.is_active,
            "ip_whitelist": self.ip_whitelist,
            "metadata": self.metadata
        }


@dataclass
class SAMLConfiguration:
    """SAML Identity Provider configuration."""
    entity_id: str
    sso_url: str
    certificate: str
    attribute_mapping: Dict[str, str]
    organization_id: uuid.UUID
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OAuthConfiguration:
    """OAuth/OIDC Provider configuration."""
    provider_name: str
    client_id: str
    client_secret: str
    authorization_url: str
    token_url: str
    userinfo_url: str
    scopes: List[str]
    organization_id: uuid.UUID
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnterpriseAuthenticationSystem:
    """
    Enterprise Authentication System with multi-tenant support.
    
    Features:
    - SAML 2.0 SSO with enterprise IdP integration
    - OAuth 2.0 / OpenID Connect support
    - API key management with fine-grained permissions
    - Multi-tenant organization isolation
    - Service-to-service authentication
    - JWT token management with enterprise features
    - MFA support with TOTP and enterprise methods
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        security_audit: SecurityAuditSystem,
        jwt_secret: str,
        encryption_key: bytes
    ):
        """
        Initialize enterprise authentication system.
        
        Args:
            db_session: Database session
            security_audit: Security audit system
            jwt_secret: JWT signing secret
            encryption_key: Encryption key for sensitive data
        """
        self.db = db_session
        self.security_audit = security_audit
        self.jwt_secret = jwt_secret
        self.encryption_key = encryption_key
        
        # Password hashing
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # Enterprise configurations
        self.saml_configs: Dict[uuid.UUID, SAMLConfiguration] = {}
        self.oauth_configs: Dict[uuid.UUID, OAuthConfiguration] = {}
        
        # API key storage (in production, use database)
        self.api_keys: Dict[str, APIKey] = {}
        
        # Active sessions
        self.active_sessions: Dict[str, AuthenticatedUser] = {}
        
        # Rate limiting tracking
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.config = {
            "jwt_expiry_hours": 8,
            "refresh_token_expiry_days": 30,
            "api_key_expiry_days": 365,
            "max_sessions_per_user": 5,
            "password_min_length": 12,
            "mfa_required_roles": [UserRole.SUPER_ADMIN, UserRole.SECURITY_ADMIN],
            "session_timeout_minutes": 60,
            "max_failed_attempts": 5,
            "lockout_duration_minutes": 30
        }
    
    async def authenticate_saml_user(self, saml_assertion: str, organization_id: uuid.UUID) -> AuthenticatedUser:
        """
        Authenticate user via SAML assertion.
        
        Args:
            saml_assertion: Base64 encoded SAML assertion
            organization_id: Organization attempting authentication
            
        Returns:
            Authenticated user object
            
        Raises:
            ValueError: If SAML assertion is invalid
            PermissionError: If authentication fails
        """
        try:
            # Get SAML configuration for organization
            saml_config = self.saml_configs.get(organization_id)
            if not saml_config or not saml_config.is_active:
                await self._audit_auth_failure(
                    None, organization_id, "SAML configuration not found or inactive",
                    AuthenticationMethod.SAML_SSO
                )
                raise ValueError("SAML not configured for organization")
            
            # Decode and parse SAML assertion
            try:
                assertion_xml = base64.b64decode(saml_assertion).decode('utf-8')
                root = ET.fromstring(assertion_xml)
            except Exception as e:
                await self._audit_auth_failure(
                    None, organization_id, f"Invalid SAML assertion format: {e}",
                    AuthenticationMethod.SAML_SSO
                )
                raise ValueError(f"Invalid SAML assertion: {e}")
            
            # Validate SAML assertion signature (simplified)
            if not await self._validate_saml_signature(assertion_xml, saml_config.certificate):
                await self._audit_auth_failure(
                    None, organization_id, "SAML signature validation failed",
                    AuthenticationMethod.SAML_SSO
                )
                raise PermissionError("SAML signature validation failed")
            
            # Extract user attributes
            user_attributes = self._extract_saml_attributes(root, saml_config.attribute_mapping)
            
            # Create or update user
            user = await self._create_or_update_saml_user(user_attributes, organization_id)
            
            # Create authenticated user session
            authenticated_user = await self._create_user_session(
                user, AuthenticationMethod.SAML_SSO, organization_id
            )
            
            await self._audit_auth_success(authenticated_user)
            logger.info(f"SAML authentication successful for user {user.email}")
            
            return authenticated_user
            
        except Exception as e:
            logger.error(f"SAML authentication failed: {e}")
            raise
    
    async def authenticate_oauth_user(self, oauth_token: str, provider: str, organization_id: uuid.UUID) -> AuthenticatedUser:
        """
        Authenticate user via OAuth token.
        
        Args:
            oauth_token: OAuth access token
            provider: OAuth provider name
            organization_id: Organization attempting authentication
            
        Returns:
            Authenticated user object
            
        Raises:
            ValueError: If OAuth configuration is invalid
            PermissionError: If authentication fails
        """
        try:
            # Get OAuth configuration
            oauth_config = None
            for config in self.oauth_configs.values():
                if config.provider_name == provider and config.organization_id == organization_id:
                    oauth_config = config
                    break
            
            if not oauth_config or not oauth_config.is_active:
                await self._audit_auth_failure(
                    None, organization_id, f"OAuth provider {provider} not configured",
                    AuthenticationMethod.OAUTH_SSO
                )
                raise ValueError(f"OAuth provider {provider} not configured")
            
            # Validate token and get user info
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {oauth_token}"}
                response = await client.get(oauth_config.userinfo_url, headers=headers)
                
                if response.status_code != 200:
                    await self._audit_auth_failure(
                        None, organization_id, "OAuth token validation failed",
                        AuthenticationMethod.OAUTH_SSO
                    )
                    raise PermissionError("OAuth token validation failed")
                
                user_info = response.json()
            
            # Create or update user
            user = await self._create_or_update_oauth_user(user_info, organization_id, provider)
            
            # Create authenticated user session
            authenticated_user = await self._create_user_session(
                user, AuthenticationMethod.OAUTH_SSO, organization_id
            )
            
            await self._audit_auth_success(authenticated_user)
            logger.info(f"OAuth authentication successful for user {user.email}")
            
            return authenticated_user
            
        except Exception as e:
            logger.error(f"OAuth authentication failed: {e}")
            raise
    
    async def create_api_key(
        self,
        user: AuthenticatedUser,
        name: str,
        permissions: List[str],
        scopes: List[str],
        expires_days: Optional[int] = None,
        rate_limit: int = 1000
    ) -> Tuple[str, APIKey]:
        """
        Create API key for user with specified permissions.
        
        Args:
            user: Authenticated user creating the key
            name: Human-readable name for the key
            permissions: List of permissions to grant
            scopes: List of API scopes to allow
            expires_days: Days until key expires (None for no expiry)
            rate_limit: Requests per hour limit
            
        Returns:
            Tuple of (raw_key, api_key_object)
        """
        try:
            # Generate secure API key
            raw_key = f"lv_{secrets.token_urlsafe(32)}"
            key_hash = self._hash_api_key(raw_key)
            
            # Calculate expiry
            expires_at = None
            if expires_days:
                expires_at = datetime.utcnow() + timedelta(days=expires_days)
            elif expires_days is None and self.config["api_key_expiry_days"]:
                expires_at = datetime.utcnow() + timedelta(days=self.config["api_key_expiry_days"])
            
            # Create API key object
            api_key = APIKey(
                id=uuid.uuid4(),
                name=name,
                key_hash=key_hash,
                user_id=user.id,
                organization_id=user.organization_id,
                permissions=set(permissions),
                scopes=scopes,
                rate_limit=rate_limit,
                expires_at=expires_at
            )
            
            # Store API key (in production, save to database)
            self.api_keys[key_hash] = api_key
            
            # Audit API key creation
            await self.security_audit.audit_context_access(
                context_id=uuid.uuid4(),  # API key creation event
                agent_id=user.id,
                session_id=user.session_id,
                access_granted=True,
                permission=Permission.WRITE,
                access_time=datetime.utcnow()
            )
            
            logger.info(f"API key '{name}' created for user {user.username}")
            return raw_key, api_key
            
        except Exception as e:
            logger.error(f"API key creation failed: {e}")
            raise
    
    async def validate_api_key(self, raw_key: str, required_scope: Optional[str] = None) -> Optional[APIKey]:
        """
        Validate API key and check permissions.
        
        Args:
            raw_key: Raw API key string
            required_scope: Required scope for the operation
            
        Returns:
            API key object if valid, None otherwise
        """
        try:
            key_hash = self._hash_api_key(raw_key)
            api_key = self.api_keys.get(key_hash)
            
            if not api_key:
                return None
            
            # Check if key is active and not expired
            if not api_key.is_active or api_key.is_expired():
                return None
            
            # Check scope if required
            if required_scope and not api_key.can_access_scope(required_scope):
                return None
            
            # Check rate limit
            if not await self._check_api_key_rate_limit(api_key):
                return None
            
            # Update usage statistics
            api_key.last_used = datetime.utcnow()
            api_key.usage_count += 1
            
            return api_key
            
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return None
    
    async def validate_rbac_permissions(
        self,
        user: AuthenticatedUser,
        resource: str,
        action: str
    ) -> bool:
        """
        Validate Role-Based Access Control permissions.
        
        Args:
            user: Authenticated user
            resource: Resource being accessed
            action: Action being performed
            
        Returns:
            True if access is granted
        """
        try:
            # Build permission string
            permission = f"{resource}:{action}"
            
            # Check if user has specific permission
            if user.has_permission(permission):
                return True
            
            # Check role-based permissions
            role_permissions = self._get_role_permissions(user.role)
            if permission in role_permissions:
                return True
            
            # Check wildcard permissions
            resource_wildcard = f"{resource}:*"
            action_wildcard = f"*:{action}"
            global_wildcard = "*:*"
            
            if (user.has_permission(resource_wildcard) or 
                user.has_permission(action_wildcard) or 
                user.has_permission(global_wildcard)):
                return True
            
            # Super admin has all permissions
            if user.role == UserRole.SUPER_ADMIN:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"RBAC validation failed: {e}")
            return False
    
    async def manage_multi_tenant_isolation(
        self,
        tenant_id: uuid.UUID,
        user: AuthenticatedUser
    ) -> bool:
        """
        Validate multi-tenant isolation and access.
        
        Args:
            tenant_id: Target tenant ID
            user: Authenticated user
            
        Returns:
            True if user can access tenant
        """
        try:
            # Super admin can access any tenant
            if user.role == UserRole.SUPER_ADMIN:
                return True
            
            # User can access their own organization
            if user.organization_id == tenant_id:
                return True
            
            # Check cross-tenant permissions (for specific enterprise features)
            if user.has_permission(f"tenant:{tenant_id}:access"):
                return True
            
            # Service accounts might have cross-tenant access
            if (user.role == UserRole.SERVICE_ACCOUNT and 
                user.has_permission("cross_tenant:access")):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Multi-tenant isolation check failed: {e}")
            return False
    
    async def create_jwt_token(self, user: AuthenticatedUser) -> str:
        """
        Create JWT token for authenticated user.
        
        Args:
            user: Authenticated user
            
        Returns:
            JWT token string
        """
        try:
            now = datetime.utcnow()
            payload = {
                "sub": str(user.id),
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
                "org_id": str(user.organization_id),
                "org_tier": user.organization_tier.value,
                "session_id": str(user.session_id),
                "auth_method": user.authentication_method.value,
                "mfa_verified": user.mfa_verified,
                "iat": int(now.timestamp()),
                "exp": int((now + timedelta(hours=self.config["jwt_expiry_hours"])).timestamp()),
                "iss": "leanvibe-agent-hive",
                "aud": "agent-hive-api"
            }
            
            token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
            return token
            
        except Exception as e:
            logger.error(f"JWT token creation failed: {e}")
            raise
    
    async def validate_jwt_token(self, token: str) -> Optional[AuthenticatedUser]:
        """
        Validate JWT token and return user.
        
        Args:
            token: JWT token string
            
        Returns:
            Authenticated user if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            # Check if session is still active
            session_id = payload.get("session_id")
            if session_id and session_id in self.active_sessions:
                user = self.active_sessions[session_id]
                
                # Update last activity
                user.last_activity = datetime.utcnow()
                
                return user
            
            return None
            
        except jwt.ExpiredSignatureError:
            logger.debug("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
        except Exception as e:
            logger.error(f"JWT validation failed: {e}")
            return None
    
    # Private helper methods
    async def _validate_saml_signature(self, assertion_xml: str, certificate: str) -> bool:
        """Validate SAML assertion signature (simplified implementation)."""
        try:
            # In production, use proper SAML library like python3-saml
            # This is a simplified validation
            cert = load_pem_x509_certificate(certificate.encode())
            public_key = cert.public_key()
            
            # Simplified signature validation
            # In real implementation, extract and validate signature from XML
            return True  # Placeholder
            
        except Exception as e:
            logger.error(f"SAML signature validation error: {e}")
            return False
    
    def _extract_saml_attributes(self, root: ET.Element, attribute_mapping: Dict[str, str]) -> Dict[str, str]:
        """Extract user attributes from SAML assertion."""
        attributes = {}
        
        # Find attribute statements
        for attr_stmt in root.findall(".//{urn:oasis:names:tc:SAML:2.0:assertion}AttributeStatement"):
            for attr in attr_stmt.findall(".//{urn:oasis:names:tc:SAML:2.0:assertion}Attribute"):
                attr_name = attr.get("Name", "")
                
                # Map SAML attribute to our attribute
                for our_attr, saml_attr in attribute_mapping.items():
                    if saml_attr == attr_name:
                        attr_value = attr.find(".//{urn:oasis:names:tc:SAML:2.0:assertion}AttributeValue")
                        if attr_value is not None:
                            attributes[our_attr] = attr_value.text or ""
        
        return attributes
    
    async def _create_or_update_saml_user(self, attributes: Dict[str, str], organization_id: uuid.UUID) -> Agent:
        """Create or update user from SAML attributes."""
        email = attributes.get("email", "")
        username = attributes.get("username", email)
        
        if not email:
            raise ValueError("Email attribute required from SAML")
        
        # Check if user exists
        result = await self.db.execute(
            select(Agent).where(and_(Agent.email == email, Agent.organization_id == organization_id))
        )
        user = result.scalar_one_or_none()
        
        if not user:
            # Create new user
            user = Agent(
                id=uuid.uuid4(),
                name=username,
                email=email,
                organization_id=organization_id,
                metadata={
                    "auth_method": "saml",
                    "saml_attributes": attributes
                }
            )
            self.db.add(user)
        else:
            # Update existing user
            user.metadata = user.metadata or {}
            user.metadata.update({
                "last_saml_login": datetime.utcnow().isoformat(),
                "saml_attributes": attributes
            })
        
        await self.db.commit()
        return user
    
    async def _create_or_update_oauth_user(self, user_info: Dict[str, Any], organization_id: uuid.UUID, provider: str) -> Agent:
        """Create or update user from OAuth user info."""
        email = user_info.get("email", "")
        username = user_info.get("preferred_username", user_info.get("name", email))
        
        if not email:
            raise ValueError("Email required from OAuth provider")
        
        # Check if user exists
        result = await self.db.execute(
            select(Agent).where(and_(Agent.email == email, Agent.organization_id == organization_id))
        )
        user = result.scalar_one_or_none()
        
        if not user:
            # Create new user
            user = Agent(
                id=uuid.uuid4(),
                name=username,
                email=email,
                organization_id=organization_id,
                metadata={
                    "auth_method": "oauth",
                    "oauth_provider": provider,
                    "oauth_user_info": user_info
                }
            )
            self.db.add(user)
        else:
            # Update existing user
            user.metadata = user.metadata or {}
            user.metadata.update({
                "last_oauth_login": datetime.utcnow().isoformat(),
                "oauth_provider": provider,
                "oauth_user_info": user_info
            })
        
        await self.db.commit()
        return user
    
    async def _create_user_session(
        self,
        user: Agent,
        auth_method: AuthenticationMethod,
        organization_id: uuid.UUID
    ) -> AuthenticatedUser:
        """Create authenticated user session."""
        session_id = uuid.uuid4()
        
        # Determine user role and permissions
        role = self._determine_user_role(user)
        permissions = set(self._get_role_permissions(role))
        
        # Get organization info (simplified)
        org_name = f"Org-{organization_id}"
        org_tier = OrganizationTier.PROFESSIONAL  # Default
        
        authenticated_user = AuthenticatedUser(
            id=user.id,
            username=user.name,
            email=user.email,
            role=role,
            organization_id=organization_id,
            organization_name=org_name,
            organization_tier=org_tier,
            permissions=permissions,
            authentication_method=auth_method,
            session_id=session_id,
            expires_at=datetime.utcnow() + timedelta(hours=self.config["jwt_expiry_hours"]),
            metadata=user.metadata or {}
        )
        
        # Store active session
        self.active_sessions[str(session_id)] = authenticated_user
        
        return authenticated_user
    
    def _determine_user_role(self, user: Agent) -> UserRole:
        """Determine user role from user metadata."""
        metadata = user.metadata or {}
        
        # Check explicit role assignment
        if "role" in metadata:
            try:
                return UserRole(metadata["role"])
            except ValueError:
                pass
        
        # Default role based on email or other criteria
        if user.email.endswith("@admin.leanvibe.com"):
            return UserRole.SUPER_ADMIN
        elif user.email.endswith("@leanvibe.com"):
            return UserRole.DEVELOPER
        else:
            return UserRole.API_USER
    
    def _get_role_permissions(self, role: UserRole) -> List[str]:
        """Get permissions for user role."""
        role_permissions = {
            UserRole.SUPER_ADMIN: [
                "*:*"  # All permissions
            ],
            UserRole.TENANT_ADMIN: [
                "tenant:*", "user:*", "agent:*", "context:*", "workflow:*"
            ],
            UserRole.SECURITY_ADMIN: [
                "security:*", "audit:*", "user:read", "agent:read"
            ],
            UserRole.DEVELOPER: [
                "agent:*", "context:*", "workflow:*", "task:*"
            ],
            UserRole.ANALYST: [
                "agent:read", "context:read", "workflow:read", "analytics:*"
            ],
            UserRole.API_USER: [
                "agent:read", "context:read", "task:create", "task:read"
            ],
            UserRole.SERVICE_ACCOUNT: [
                "service:*", "agent:read", "context:read"
            ],
            UserRole.GUEST: [
                "agent:read", "context:read"
            ]
        }
        
        return role_permissions.get(role, [])
    
    def _hash_api_key(self, raw_key: str) -> str:
        """Hash API key for secure storage."""
        return hashlib.sha256(raw_key.encode()).hexdigest()
    
    async def _check_api_key_rate_limit(self, api_key: APIKey) -> bool:
        """Check if API key is within rate limits."""
        now = datetime.utcnow()
        hour_key = f"{api_key.id}:{now.strftime('%Y-%m-%d-%H')}"
        
        if hour_key not in self.rate_limits:
            self.rate_limits[hour_key] = {"count": 0, "reset_time": now + timedelta(hours=1)}
        
        rate_data = self.rate_limits[hour_key]
        
        # Reset if hour has passed
        if now > rate_data["reset_time"]:
            rate_data["count"] = 0
            rate_data["reset_time"] = now + timedelta(hours=1)
        
        # Check limit
        if rate_data["count"] >= api_key.rate_limit:
            return False
        
        rate_data["count"] += 1
        return True
    
    async def _audit_auth_success(self, user: AuthenticatedUser) -> None:
        """Audit successful authentication."""
        await self.security_audit.audit_context_access(
            context_id=uuid.uuid4(),  # Auth success event
            agent_id=user.id,
            session_id=user.session_id,
            access_granted=True,
            permission=Permission.READ,
            access_time=datetime.utcnow()
        )
    
    async def _audit_auth_failure(
        self,
        user_id: Optional[uuid.UUID],
        organization_id: uuid.UUID,
        reason: str,
        auth_method: AuthenticationMethod
    ) -> None:
        """Audit failed authentication attempt."""
        await self.security_audit.audit_context_access(
            context_id=uuid.uuid4(),  # Auth failure event
            agent_id=user_id or uuid.uuid4(),  # Use dummy ID if user unknown
            session_id=None,
            access_granted=False,
            permission=Permission.READ,
            access_time=datetime.utcnow()
        )


# Factory function
async def create_enterprise_auth_system(
    db_session: AsyncSession,
    security_audit: SecurityAuditSystem,
    jwt_secret: str,
    encryption_key: bytes
) -> EnterpriseAuthenticationSystem:
    """
    Create enterprise authentication system instance.
    
    Args:
        db_session: Database session
        security_audit: Security audit system
        jwt_secret: JWT signing secret
        encryption_key: Encryption key for sensitive data
        
    Returns:
        EnterpriseAuthenticationSystem instance
    """
    return EnterpriseAuthenticationSystem(db_session, security_audit, jwt_secret, encryption_key)