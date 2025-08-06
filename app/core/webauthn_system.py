"""
WebAuthn Biometric Authentication System for LeanVibe Agent Hive.

Implements FIDO2/WebAuthn standard for passwordless authentication using:
- Hardware security keys (YubiKey, etc.)
- Platform authenticators (TouchID, FaceID, Windows Hello)
- Biometric authentication
- Multi-device support with credential management

Enterprise-grade security with comprehensive audit logging and compliance features.
"""

import os
import json
import base64
import secrets
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

import structlog
from fastapi import HTTPException, Request, Response, Depends, status, APIRouter
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, update
from webauthn import generate_registration_options, verify_registration_response
from webauthn import generate_authentication_options, verify_authentication_response
from webauthn.helpers.structs import (
    AttestationConveyancePreference, AuthenticatorSelectionCriteria, 
    UserVerificationRequirement, ResidentKeyRequirement,
    PublicKeyCredentialDescriptor, AuthenticatorTransport
)
from webauthn.helpers.cose import COSEAlgorithmIdentifier
from webauthn.helpers.decode_client_data_json import decode_client_data_json
from webauthn.helpers.generate_challenge import generate_challenge

from .database import get_session
from .auth import get_auth_service
from ..models.security import AgentIdentity, SecurityAuditLog, SecurityEvent
from ..schemas.security import SecurityError

logger = structlog.get_logger()

# WebAuthn Configuration
WEBAUTHN_CONFIG = {
    "rp_id": os.getenv("WEBAUTHN_RP_ID", "localhost"),  # Relying Party ID
    "rp_name": os.getenv("WEBAUTHN_RP_NAME", "LeanVibe Agent Hive"),
    "origin": os.getenv("WEBAUTHN_ORIGIN", "http://localhost:8000"),
    "timeout": int(os.getenv("WEBAUTHN_TIMEOUT", "60000")),  # 60 seconds
    "challenge_size": 32,  # bytes
    "user_verification": UserVerificationRequirement.PREFERRED,
    "authenticator_selection": {
        "authenticator_attachment": None,  # Allow both platform and cross-platform
        "resident_key": ResidentKeyRequirement.PREFERRED,
        "user_verification": UserVerificationRequirement.PREFERRED
    },
    "supported_algorithms": [
        COSEAlgorithmIdentifier.ECDSA_SHA_256,
        COSEAlgorithmIdentifier.RSASSA_PKCS1_v1_5_SHA_256,
        COSEAlgorithmIdentifier.RSA_PSS_SHA_256,
    ],
    "attestation_preference": AttestationConveyancePreference.NONE
}


class AuthenticatorType(Enum):
    """Types of WebAuthn authenticators."""
    PLATFORM = "platform"  # TouchID, FaceID, Windows Hello
    CROSS_PLATFORM = "cross-platform"  # Security keys, YubiKey
    UNKNOWN = "unknown"


class CredentialTransport(Enum):
    """WebAuthn credential transport methods."""
    USB = "usb"
    NFC = "nfc" 
    BLE = "ble"
    INTERNAL = "internal"
    HYBRID = "hybrid"


@dataclass
class WebAuthnCredential:
    """WebAuthn credential information."""
    credential_id: str
    public_key: bytes
    sign_count: int
    user_id: str
    user_handle: bytes
    aaguid: Optional[str] = None
    authenticator_type: AuthenticatorType = AuthenticatorType.UNKNOWN
    transports: List[CredentialTransport] = field(default_factory=list)
    backup_eligible: bool = False
    backup_state: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    device_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "credential_id": self.credential_id,
            "public_key": base64.b64encode(self.public_key).decode() if self.public_key else None,
            "sign_count": self.sign_count,
            "user_id": self.user_id,
            "user_handle": base64.b64encode(self.user_handle).decode() if self.user_handle else None,
            "aaguid": self.aaguid,
            "authenticator_type": self.authenticator_type.value,
            "transports": [t.value for t in self.transports],
            "backup_eligible": self.backup_eligible,
            "backup_state": self.backup_state,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "device_name": self.device_name
        }


class RegistrationRequest(BaseModel):
    """WebAuthn registration request."""
    username: str = Field(..., min_length=1, max_length=255)
    display_name: str = Field(..., min_length=1, max_length=255)
    user_id: Optional[str] = Field(None, max_length=255)
    authenticator_selection: Optional[Dict[str, Any]] = None
    attestation: Optional[str] = Field(None, pattern="^(none|indirect|direct|enterprise)$")
    exclude_credentials: List[str] = Field(default_factory=list)
    
    @validator('username')
    def validate_username(cls, v):
        """Validate username format."""
        import re
        if not re.match(r'^[a-zA-Z0-9._-]+$', v):
            raise ValueError('Username can only contain letters, numbers, dots, underscores, and hyphens')
        return v


class RegistrationResponse(BaseModel):
    """WebAuthn registration response."""
    id: str
    rawId: str
    response: Dict[str, Any]
    type: str = "public-key"
    authenticatorAttachment: Optional[str] = None
    clientExtensionResults: Optional[Dict[str, Any]] = None


class AuthenticationRequest(BaseModel):
    """WebAuthn authentication request."""
    username: Optional[str] = Field(None, max_length=255)
    user_verification: Optional[str] = Field(None, pattern="^(required|preferred|discouraged)$")
    allowed_credentials: List[str] = Field(default_factory=list)


class AuthenticationResponse(BaseModel):
    """WebAuthn authentication response."""
    id: str
    rawId: str
    response: Dict[str, Any]
    type: str = "public-key"
    authenticatorAttachment: Optional[str] = None
    clientExtensionResults: Optional[Dict[str, Any]] = None


class WebAuthnChallenge(BaseModel):
    """WebAuthn challenge data."""
    challenge: str
    user_id: str
    username: str
    timeout: int
    created_at: datetime
    used: bool = False
    
    class Config:
        arbitrary_types_allowed = True


class CredentialInfo(BaseModel):
    """Credential information response."""
    id: str
    type: str
    transports: List[str]
    authenticator_type: str
    device_name: Optional[str]
    created_at: str
    last_used: Optional[str]
    backup_eligible: bool
    backup_state: bool


class WebAuthnSystem:
    """
    Comprehensive WebAuthn Authentication System.
    
    Provides enterprise-grade passwordless authentication with:
    - FIDO2/WebAuthn standard compliance
    - Support for platform and cross-platform authenticators  
    - Hardware security key integration
    - Biometric authentication (TouchID, FaceID, Windows Hello)
    - Multi-device credential management
    - Comprehensive security auditing
    """
    
    def __init__(self, db_session: AsyncSession):
        """
        Initialize WebAuthn System.
        
        Args:
            db_session: Database session for credential storage
        """
        self.db = db_session
        self.config = WEBAUTHN_CONFIG.copy()
        
        # Active challenges storage (in production, use Redis)
        self.active_challenges: Dict[str, WebAuthnChallenge] = {}
        
        # Credential storage (in-memory for demo, should use database)
        self.credentials: Dict[str, List[WebAuthnCredential]] = {}
        
        # Performance metrics
        self.metrics = {
            "registrations_initiated": 0,
            "registrations_completed": 0,
            "registrations_failed": 0,
            "authentications_initiated": 0,
            "authentications_completed": 0,
            "authentications_failed": 0,
            "credentials_created": 0,
            "credentials_revoked": 0,
            "avg_auth_time_ms": 0.0,
            "authenticator_types": {},
            "transport_usage": {},
            "error_counts": {}
        }
        
        logger.info("WebAuthn System initialized", 
                   rp_id=self.config["rp_id"],
                   rp_name=self.config["rp_name"])
    
    async def initiate_registration(self, request: RegistrationRequest) -> Dict[str, Any]:
        """
        Initiate WebAuthn credential registration.
        
        Args:
            request: Registration request parameters
            
        Returns:
            Registration options for client
        """
        try:
            # Generate user ID if not provided
            user_id = request.user_id or str(uuid.uuid4())
            user_handle = user_id.encode('utf-8')
            
            # Get existing credentials to exclude
            existing_creds = self.credentials.get(request.username, [])
            exclude_credentials = [
                PublicKeyCredentialDescriptor(
                    id=base64.b64decode(cred.credential_id.encode()),
                    transports=[AuthenticatorTransport(t.value) for t in cred.transports]
                )
                for cred in existing_creds
            ]
            
            # Add explicitly excluded credentials
            for cred_id in request.exclude_credentials:
                try:
                    exclude_credentials.append(
                        PublicKeyCredentialDescriptor(
                            id=base64.b64decode(cred_id.encode())
                        )
                    )
                except Exception:
                    pass  # Ignore invalid credential IDs
            
            # Configure authenticator selection
            authenticator_selection = AuthenticatorSelectionCriteria(
                authenticator_attachment=None,  # Allow both platform and cross-platform
                resident_key=ResidentKeyRequirement.PREFERRED,
                user_verification=UserVerificationRequirement.PREFERRED
            )
            
            if request.authenticator_selection:
                if "authenticatorAttachment" in request.authenticator_selection:
                    authenticator_selection.authenticator_attachment = request.authenticator_selection["authenticatorAttachment"]
                
                if "residentKey" in request.authenticator_selection:
                    authenticator_selection.resident_key = ResidentKeyRequirement(
                        request.authenticator_selection["residentKey"]
                    )
                
                if "userVerification" in request.authenticator_selection:
                    authenticator_selection.user_verification = UserVerificationRequirement(
                        request.authenticator_selection["userVerification"]
                    )
            
            # Generate registration options
            options = generate_registration_options(
                rp_id=self.config["rp_id"],
                rp_name=self.config["rp_name"],
                user_id=user_handle,
                user_name=request.username,
                user_display_name=request.display_name,
                attestation=AttestationConveyancePreference(
                    request.attestation or "none"
                ),
                authenticator_selection=authenticator_selection,
                challenge=generate_challenge(self.config["challenge_size"]),
                exclude_credentials=exclude_credentials,
                supported_pub_key_algs=self.config["supported_algorithms"],
                timeout=self.config["timeout"]
            )
            
            # Store challenge
            challenge_key = base64.b64encode(options.challenge).decode()
            self.active_challenges[challenge_key] = WebAuthnChallenge(
                challenge=challenge_key,
                user_id=user_id,
                username=request.username,
                timeout=self.config["timeout"],
                created_at=datetime.utcnow()
            )
            
            # Update metrics
            self.metrics["registrations_initiated"] += 1
            
            # Log registration initiation
            await self._log_webauthn_event(
                action="initiate_registration",
                username=request.username,
                success=True,
                metadata={
                    "user_id": user_id,
                    "display_name": request.display_name,
                    "authenticator_selection": request.authenticator_selection,
                    "excluded_credentials": len(exclude_credentials)
                }
            )
            
            return {
                "publicKey": {
                    "challenge": base64.b64encode(options.challenge).decode(),
                    "rp": {
                        "name": options.rp.name,
                        "id": options.rp.id
                    },
                    "user": {
                        "id": base64.b64encode(options.user.id).decode(),
                        "name": options.user.name,
                        "displayName": options.user.display_name
                    },
                    "pubKeyCredParams": [
                        {"alg": alg.alg, "type": "public-key"}
                        for alg in options.pub_key_cred_params
                    ],
                    "timeout": options.timeout,
                    "excludeCredentials": [
                        {
                            "id": base64.b64encode(cred.id).decode(),
                            "type": "public-key",
                            "transports": [t.value for t in (cred.transports or [])]
                        }
                        for cred in (options.exclude_credentials or [])
                    ],
                    "authenticatorSelection": {
                        "authenticatorAttachment": authenticator_selection.authenticator_attachment,
                        "residentKey": authenticator_selection.resident_key.value,
                        "userVerification": authenticator_selection.user_verification.value
                    },
                    "attestation": options.attestation.value
                }
            }
            
        except Exception as e:
            logger.error("Registration initiation failed", error=str(e))
            self.metrics["registrations_failed"] += 1
            self.metrics["error_counts"]["registration_init_error"] = (
                self.metrics["error_counts"].get("registration_init_error", 0) + 1
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Registration initiation failed: {str(e)}"
            )
    
    async def complete_registration(self, 
                                   credential_response: RegistrationResponse,
                                   username: str) -> Dict[str, Any]:
        """
        Complete WebAuthn credential registration.
        
        Args:
            credential_response: Client's credential response
            username: Username for registration
            
        Returns:
            Registration result with credential info
        """
        try:
            # Decode challenge
            challenge_key = credential_response.response.get("clientDataJSON", "")
            client_data = decode_client_data_json(
                base64.b64decode(challenge_key.encode())
            )
            challenge = base64.b64encode(client_data.challenge).decode()
            
            # Validate challenge
            stored_challenge = self.active_challenges.get(challenge)
            if not stored_challenge or stored_challenge.used:
                raise ValueError("Invalid or expired challenge")
            
            if stored_challenge.username != username:
                raise ValueError("Username mismatch")
            
            # Check challenge expiration
            if (datetime.utcnow() - stored_challenge.created_at).total_seconds() * 1000 > stored_challenge.timeout:
                raise ValueError("Challenge expired")
            
            # Verify registration response
            verification = verify_registration_response(
                credential={
                    "id": credential_response.id,
                    "rawId": credential_response.rawId,
                    "response": {
                        "attestationObject": credential_response.response["attestationObject"],
                        "clientDataJSON": credential_response.response["clientDataJSON"]
                    },
                    "type": credential_response.type,
                    "authenticatorAttachment": credential_response.authenticatorAttachment,
                    "clientExtensionResults": credential_response.clientExtensionResults or {}
                },
                expected_challenge=base64.b64decode(challenge.encode()),
                expected_origin=self.config["origin"],
                expected_rp_id=self.config["rp_id"],
                require_user_verification=False  # Set based on policy
            )
            
            if not verification.verified:
                raise ValueError("Registration verification failed")
            
            # Determine authenticator type
            authenticator_type = AuthenticatorType.UNKNOWN
            if credential_response.authenticatorAttachment == "platform":
                authenticator_type = AuthenticatorType.PLATFORM
            elif credential_response.authenticatorAttachment == "cross-platform":
                authenticator_type = AuthenticatorType.CROSS_PLATFORM
            
            # Determine transports
            transports = []
            if hasattr(verification.credential, 'transports') and verification.credential.transports:
                for transport in verification.credential.transports:
                    try:
                        transports.append(CredentialTransport(transport.value))
                    except ValueError:
                        pass  # Ignore unknown transports
            
            # Create credential record
            credential = WebAuthnCredential(
                credential_id=base64.b64encode(verification.credential_id).decode(),
                public_key=verification.credential_public_key,
                sign_count=verification.sign_count,
                user_id=stored_challenge.user_id,
                user_handle=stored_challenge.user_id.encode('utf-8'),
                aaguid=str(verification.aaguid) if verification.aaguid else None,
                authenticator_type=authenticator_type,
                transports=transports,
                backup_eligible=getattr(verification, 'credential_backup_eligible', False),
                backup_state=getattr(verification, 'credential_backup_state', False),
                device_name=self._generate_device_name(authenticator_type, transports)
            )
            
            # Store credential
            if username not in self.credentials:
                self.credentials[username] = []
            self.credentials[username].append(credential)
            
            # Mark challenge as used
            stored_challenge.used = True
            
            # Update metrics
            self.metrics["registrations_completed"] += 1
            self.metrics["credentials_created"] += 1
            
            auth_type_key = authenticator_type.value
            self.metrics["authenticator_types"][auth_type_key] = (
                self.metrics["authenticator_types"].get(auth_type_key, 0) + 1
            )
            
            for transport in transports:
                transport_key = transport.value
                self.metrics["transport_usage"][transport_key] = (
                    self.metrics["transport_usage"].get(transport_key, 0) + 1
                )
            
            # Log successful registration
            await self._log_webauthn_event(
                action="complete_registration", 
                username=username,
                success=True,
                metadata={
                    "credential_id": credential.credential_id,
                    "authenticator_type": authenticator_type.value,
                    "transports": [t.value for t in transports],
                    "backup_eligible": credential.backup_eligible,
                    "device_name": credential.device_name
                }
            )
            
            return {
                "verified": True,
                "credential": {
                    "id": credential.credential_id,
                    "type": "public-key",
                    "authenticator_type": authenticator_type.value,
                    "transports": [t.value for t in transports],
                    "device_name": credential.device_name,
                    "backup_eligible": credential.backup_eligible,
                    "created_at": credential.created_at.isoformat()
                }
            }
            
        except Exception as e:
            logger.error("Registration completion failed", username=username, error=str(e))
            self.metrics["registrations_failed"] += 1
            self.metrics["error_counts"]["registration_complete_error"] = (
                self.metrics["error_counts"].get("registration_complete_error", 0) + 1
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Registration completion failed: {str(e)}"
            )
    
    async def initiate_authentication(self, request: AuthenticationRequest) -> Dict[str, Any]:
        """
        Initiate WebAuthn authentication.
        
        Args:
            request: Authentication request parameters
            
        Returns:
            Authentication options for client
        """
        try:
            allowed_credentials = []
            
            if request.username:
                # Get credentials for specific user
                user_credentials = self.credentials.get(request.username, [])
                allowed_credentials = [
                    PublicKeyCredentialDescriptor(
                        id=base64.b64decode(cred.credential_id.encode()),
                        transports=[AuthenticatorTransport(t.value) for t in cred.transports]
                    )
                    for cred in user_credentials
                ]
            elif request.allowed_credentials:
                # Use explicitly allowed credentials
                for cred_id in request.allowed_credentials:
                    try:
                        allowed_credentials.append(
                            PublicKeyCredentialDescriptor(
                                id=base64.b64decode(cred_id.encode())
                            )
                        )
                    except Exception:
                        pass  # Ignore invalid credential IDs
            
            # Generate authentication options
            options = generate_authentication_options(
                rp_id=self.config["rp_id"],
                challenge=generate_challenge(self.config["challenge_size"]),
                allow_credentials=allowed_credentials or None,  # None allows discoverable credentials
                user_verification=UserVerificationRequirement(
                    request.user_verification or "preferred"
                ),
                timeout=self.config["timeout"]
            )
            
            # Store challenge
            challenge_key = base64.b64encode(options.challenge).decode()
            self.active_challenges[challenge_key] = WebAuthnChallenge(
                challenge=challenge_key,
                user_id="",  # Will be determined during authentication
                username=request.username or "",
                timeout=self.config["timeout"],
                created_at=datetime.utcnow()
            )
            
            # Update metrics
            self.metrics["authentications_initiated"] += 1
            
            # Log authentication initiation
            await self._log_webauthn_event(
                action="initiate_authentication",
                username=request.username or "discoverable",
                success=True,
                metadata={
                    "user_verification": request.user_verification,
                    "allowed_credentials": len(allowed_credentials),
                    "discoverable_auth": not bool(allowed_credentials)
                }
            )
            
            return {
                "publicKey": {
                    "challenge": base64.b64encode(options.challenge).decode(),
                    "timeout": options.timeout,
                    "rpId": options.rp_id,
                    "allowCredentials": [
                        {
                            "id": base64.b64encode(cred.id).decode(),
                            "type": "public-key",
                            "transports": [t.value for t in (cred.transports or [])]
                        }
                        for cred in (options.allow_credentials or [])
                    ] if options.allow_credentials else [],
                    "userVerification": options.user_verification.value
                }
            }
            
        except Exception as e:
            logger.error("Authentication initiation failed", error=str(e))
            self.metrics["authentications_failed"] += 1
            self.metrics["error_counts"]["authentication_init_error"] = (
                self.metrics["error_counts"].get("authentication_init_error", 0) + 1
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Authentication initiation failed: {str(e)}"
            )
    
    async def complete_authentication(self, 
                                      credential_response: AuthenticationResponse) -> Dict[str, Any]:
        """
        Complete WebAuthn authentication.
        
        Args:
            credential_response: Client's authentication response
            
        Returns:
            Authentication result with user info
        """
        import time
        start_time = time.time()
        
        try:
            # Decode challenge
            challenge_key = credential_response.response.get("clientDataJSON", "")
            client_data = decode_client_data_json(
                base64.b64decode(challenge_key.encode())
            )
            challenge = base64.b64encode(client_data.challenge).decode()
            
            # Validate challenge
            stored_challenge = self.active_challenges.get(challenge)
            if not stored_challenge or stored_challenge.used:
                raise ValueError("Invalid or expired challenge")
            
            # Check challenge expiration
            if (datetime.utcnow() - stored_challenge.created_at).total_seconds() * 1000 > stored_challenge.timeout:
                raise ValueError("Challenge expired")
            
            # Find credential
            credential_id = credential_response.id
            found_credential = None
            username = None
            
            # Search all users for the credential
            for user, user_credentials in self.credentials.items():
                for cred in user_credentials:
                    if cred.credential_id == credential_id:
                        found_credential = cred
                        username = user
                        break
                if found_credential:
                    break
            
            if not found_credential:
                raise ValueError("Credential not found")
            
            # Verify authentication response
            verification = verify_authentication_response(
                credential={
                    "id": credential_response.id,
                    "rawId": credential_response.rawId,
                    "response": {
                        "authenticatorData": credential_response.response["authenticatorData"],
                        "clientDataJSON": credential_response.response["clientDataJSON"],
                        "signature": credential_response.response["signature"],
                        "userHandle": credential_response.response.get("userHandle")
                    },
                    "type": credential_response.type,
                    "authenticatorAttachment": credential_response.authenticatorAttachment,
                    "clientExtensionResults": credential_response.clientExtensionResults or {}
                },
                expected_challenge=base64.b64decode(challenge.encode()),
                expected_origin=self.config["origin"],
                expected_rp_id=self.config["rp_id"],
                credential_public_key=found_credential.public_key,
                credential_current_sign_count=found_credential.sign_count,
                require_user_verification=False  # Set based on policy
            )
            
            if not verification.verified:
                raise ValueError("Authentication verification failed")
            
            # Update credential
            found_credential.sign_count = verification.new_sign_count
            found_credential.last_used = datetime.utcnow()
            
            # Mark challenge as used
            stored_challenge.used = True
            
            # Calculate authentication time
            auth_time_ms = (time.time() - start_time) * 1000
            
            # Update metrics
            self.metrics["authentications_completed"] += 1
            
            current_avg = self.metrics["avg_auth_time_ms"]
            total_auths = self.metrics["authentications_completed"]
            self.metrics["avg_auth_time_ms"] = (
                (current_avg * (total_auths - 1) + auth_time_ms) / total_auths
            )
            
            # Log successful authentication
            await self._log_webauthn_event(
                action="complete_authentication",
                username=username,
                success=True,
                metadata={
                    "credential_id": found_credential.credential_id,
                    "authenticator_type": found_credential.authenticator_type.value,
                    "sign_count": verification.new_sign_count,
                    "auth_time_ms": round(auth_time_ms, 2),
                    "user_verified": getattr(verification, 'credential_user_verified', False)
                }
            )
            
            return {
                "verified": True,
                "user": {
                    "id": found_credential.user_id,
                    "username": username,
                    "credential_id": found_credential.credential_id,
                    "authenticator_type": found_credential.authenticator_type.value,
                    "device_name": found_credential.device_name
                },
                "authentication_time_ms": round(auth_time_ms, 2)
            }
            
        except Exception as e:
            logger.error("Authentication completion failed", error=str(e))
            self.metrics["authentications_failed"] += 1
            self.metrics["error_counts"]["authentication_complete_error"] = (
                self.metrics["error_counts"].get("authentication_complete_error", 0) + 1
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Authentication failed: {str(e)}"
            )
    
    async def get_user_credentials(self, username: str) -> List[CredentialInfo]:
        """
        Get all credentials for a user.
        
        Args:
            username: Username to get credentials for
            
        Returns:
            List of credential information
        """
        user_credentials = self.credentials.get(username, [])
        
        return [
            CredentialInfo(
                id=cred.credential_id,
                type="public-key",
                transports=[t.value for t in cred.transports],
                authenticator_type=cred.authenticator_type.value,
                device_name=cred.device_name,
                created_at=cred.created_at.isoformat(),
                last_used=cred.last_used.isoformat() if cred.last_used else None,
                backup_eligible=cred.backup_eligible,
                backup_state=cred.backup_state
            )
            for cred in user_credentials
        ]
    
    async def revoke_credential(self, username: str, credential_id: str) -> bool:
        """
        Revoke a specific credential.
        
        Args:
            username: Username owning the credential
            credential_id: Credential ID to revoke
            
        Returns:
            True if credential was revoked
        """
        try:
            user_credentials = self.credentials.get(username, [])
            
            for i, cred in enumerate(user_credentials):
                if cred.credential_id == credential_id:
                    # Remove credential
                    removed_credential = user_credentials.pop(i)
                    
                    # Update metrics
                    self.metrics["credentials_revoked"] += 1
                    
                    # Log credential revocation
                    await self._log_webauthn_event(
                        action="revoke_credential",
                        username=username,
                        success=True,
                        metadata={
                            "credential_id": credential_id,
                            "device_name": removed_credential.device_name,
                            "authenticator_type": removed_credential.authenticator_type.value
                        }
                    )
                    
                    return True
            
            return False
            
        except Exception as e:
            logger.error("Credential revocation failed", username=username, credential_id=credential_id, error=str(e))
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get WebAuthn system metrics."""
        return {
            "webauthn_metrics": self.metrics.copy(),
            "active_challenges": len(self.active_challenges),
            "total_users": len(self.credentials),
            "total_credentials": sum(len(creds) for creds in self.credentials.values()),
            "config": {
                "rp_id": self.config["rp_id"],
                "rp_name": self.config["rp_name"],
                "timeout": self.config["timeout"],
                "user_verification": self.config["user_verification"].value
            }
        }
    
    def _generate_device_name(self, auth_type: AuthenticatorType, transports: List[CredentialTransport]) -> str:
        """Generate a friendly device name."""
        if auth_type == AuthenticatorType.PLATFORM:
            if CredentialTransport.INTERNAL in transports:
                return "Built-in authenticator"
            else:
                return "Platform authenticator"
        elif auth_type == AuthenticatorType.CROSS_PLATFORM:
            if CredentialTransport.USB in transports:
                return "USB Security Key"
            elif CredentialTransport.NFC in transports:
                return "NFC Security Key"
            elif CredentialTransport.BLE in transports:
                return "Bluetooth Security Key"
            else:
                return "Security Key"
        else:
            return "Unknown authenticator"
    
    async def _log_webauthn_event(self,
                                  action: str,
                                  username: str,
                                  success: bool,
                                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log WebAuthn audit event."""
        audit_log = SecurityAuditLog(
            agent_id=None,  # WebAuthn events are user-level
            human_controller=username or "anonymous",
            action=action,
            resource="webauthn_authentication",
            resource_id=username,
            success=success,
            metadata={
                "username": username,
                "webauthn_system": True,
                **(metadata or {})
            }
        )
        
        self.db.add(audit_log)
        # Note: Commit handled by caller


# Global WebAuthn system instance
_webauthn_system: Optional[WebAuthnSystem] = None


async def get_webauthn_system(db: AsyncSession = Depends(get_session)) -> WebAuthnSystem:
    """Get or create WebAuthn system instance."""
    global _webauthn_system
    if _webauthn_system is None:
        _webauthn_system = WebAuthnSystem(db)
    return _webauthn_system


# FastAPI Routes
webauthn_router = APIRouter(prefix="/webauthn", tags=["WebAuthn Authentication"])


@webauthn_router.post("/register/begin")
async def begin_registration(
    request: RegistrationRequest,
    webauthn: WebAuthnSystem = Depends(get_webauthn_system)
) -> Dict[str, Any]:
    """Begin WebAuthn registration process."""
    return await webauthn.initiate_registration(request)


@webauthn_router.post("/register/complete")
async def complete_registration(
    credential: RegistrationResponse,
    username: str,
    webauthn: WebAuthnSystem = Depends(get_webauthn_system)
) -> Dict[str, Any]:
    """Complete WebAuthn registration process."""
    return await webauthn.complete_registration(credential, username)


@webauthn_router.post("/authenticate/begin")
async def begin_authentication(
    request: AuthenticationRequest,
    webauthn: WebAuthnSystem = Depends(get_webauthn_system)
) -> Dict[str, Any]:
    """Begin WebAuthn authentication process."""
    return await webauthn.initiate_authentication(request)


@webauthn_router.post("/authenticate/complete")
async def complete_authentication(
    credential: AuthenticationResponse,
    webauthn: WebAuthnSystem = Depends(get_webauthn_system)
) -> Dict[str, Any]:
    """Complete WebAuthn authentication process."""
    return await webauthn.complete_authentication(credential)


@webauthn_router.get("/credentials/{username}")
async def get_user_credentials(
    username: str,
    webauthn: WebAuthnSystem = Depends(get_webauthn_system)
) -> List[CredentialInfo]:
    """Get all credentials for a user."""
    return await webauthn.get_user_credentials(username)


@webauthn_router.delete("/credentials/{username}/{credential_id}")
async def revoke_credential(
    username: str,
    credential_id: str,
    webauthn: WebAuthnSystem = Depends(get_webauthn_system)
) -> Dict[str, Any]:
    """Revoke a specific credential."""
    success = await webauthn.revoke_credential(username, credential_id)
    return {"revoked": success}


@webauthn_router.get("/metrics")
async def get_webauthn_metrics(
    webauthn: WebAuthnSystem = Depends(get_webauthn_system)
) -> Dict[str, Any]:
    """Get WebAuthn system metrics."""
    return webauthn.get_metrics()


# Export WebAuthn components
__all__ = [
    "WebAuthnSystem", "get_webauthn_system", "webauthn_router",
    "RegistrationRequest", "RegistrationResponse", "AuthenticationRequest", 
    "AuthenticationResponse", "CredentialInfo", "WebAuthnCredential"
]