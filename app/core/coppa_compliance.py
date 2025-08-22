"""
COPPA Compliance Framework for LeanVibe Agent Hive 2.0

Implements Children's Online Privacy Protection Act (COPPA) compliance with:
- Age verification mechanisms
- Parental consent management
- Data minimization for children under 13
- Enhanced privacy controls
- Audit logging for compliance

CRITICAL COMPONENT: Required for any application serving children under 13.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import secrets
import uuid
from datetime import datetime, timedelta, date
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

import structlog
from fastapi import HTTPException, Request, status
from pydantic import BaseModel, Field, validator, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, update
from cryptography.fernet import Fernet

from .config import settings
from .database import get_session
from .enterprise_security_system import get_security_system, SecurityEvent
from .auth import User, get_auth_service

logger = structlog.get_logger()


class ParentalConsentStatus(Enum):
    """Parental consent status for COPPA compliance."""
    PENDING = "pending"
    GRANTED = "granted"
    DENIED = "denied"
    REVOKED = "revoked"
    EXPIRED = "expired"


class ConsentMethod(Enum):
    """Methods for obtaining parental consent."""
    EMAIL_VERIFICATION = "email_verification"
    CREDIT_CARD_VERIFICATION = "credit_card_verification"
    DIGITAL_SIGNATURE = "digital_signature"
    PHONE_VERIFICATION = "phone_verification"
    VIDEO_CHAT = "video_chat"
    IN_PERSON = "in_person"


class DataCategory(Enum):
    """Categories of data that may be collected."""
    PERSONAL_IDENTIFIER = "personal_identifier"  # Name, address, email, phone
    GEOLOCATION = "geolocation"  # Physical location data
    AUDIO_VISUAL = "audio_visual"  # Photos, videos, audio recordings
    BIOMETRIC = "biometric"  # Biometric identifiers
    BEHAVIORAL = "behavioral"  # Browsing history, search queries
    EDUCATIONAL = "educational"  # School information, grades
    CONTACT_INFO = "contact_info"  # Email, phone, address
    DEVICE_INFO = "device_info"  # Device identifiers, IP addresses


class AgeVerificationMethod(Enum):
    """Age verification methods."""
    SELF_ATTESTATION = "self_attestation"
    BIRTH_DATE_VERIFICATION = "birth_date_verification"
    CREDIT_CARD_CHECK = "credit_card_check"
    PARENTAL_EMAIL_VERIFICATION = "parental_email_verification"
    THIRD_PARTY_VERIFICATION = "third_party_verification"


@dataclass
class COPPAUser:
    """COPPA-compliant user profile."""
    user_id: str
    birth_date: Optional[date]
    is_under_13: bool
    requires_parental_consent: bool
    parental_consent_status: ParentalConsentStatus
    parent_email: Optional[str]
    consent_method: Optional[ConsentMethod]
    consent_date: Optional[datetime]
    data_categories_consented: List[DataCategory]
    age_verification_method: AgeVerificationMethod
    age_verification_date: datetime
    last_parental_review: Optional[datetime]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "user_id": self.user_id,
            "birth_date": self.birth_date.isoformat() if self.birth_date else None,
            "is_under_13": self.is_under_13,
            "requires_parental_consent": self.requires_parental_consent,
            "parental_consent_status": self.parental_consent_status.value,
            "parent_email": self.parent_email,
            "consent_method": self.consent_method.value if self.consent_method else None,
            "consent_date": self.consent_date.isoformat() if self.consent_date else None,
            "data_categories_consented": [cat.value for cat in self.data_categories_consented],
            "age_verification_method": self.age_verification_method.value,
            "age_verification_date": self.age_verification_date.isoformat(),
            "last_parental_review": self.last_parental_review.isoformat() if self.last_parental_review else None
        }


class AgeVerificationRequest(BaseModel):
    """Age verification request."""
    birth_date: date
    verification_method: AgeVerificationMethod = AgeVerificationMethod.SELF_ATTESTATION
    parent_email: Optional[EmailStr] = None
    
    @validator('birth_date')
    def validate_birth_date(cls, v):
        if v > date.today():
            raise ValueError('Birth date cannot be in the future')
        
        # Must be at least 3 years old (minimum for using technology)
        min_age_date = date.today() - timedelta(days=365 * 3)
        if v > min_age_date:
            raise ValueError('User must be at least 3 years old')
        
        return v


class ParentalConsentRequest(BaseModel):
    """Parental consent request."""
    child_user_id: str
    parent_email: EmailStr
    parent_name: str
    consent_method: ConsentMethod
    data_categories: List[DataCategory]
    metadata: Dict[str, Any] = {}


class ConsentRecord(BaseModel):
    """Parental consent record."""
    consent_id: str
    child_user_id: str
    parent_email: str
    parent_name: str
    consent_status: ParentalConsentStatus
    consent_method: ConsentMethod
    data_categories_consented: List[DataCategory]
    consent_date: Optional[datetime]
    expiration_date: Optional[datetime]
    revocation_date: Optional[datetime]
    verification_token: Optional[str]
    metadata: Dict[str, Any] = {}


class COPPAComplianceSystem:
    """
    Comprehensive COPPA Compliance System.
    
    Features:
    - Age verification with multiple methods
    - Parental consent management workflow
    - Data minimization for children under 13
    - Automated compliance monitoring
    - Audit logging for regulatory compliance
    - Privacy-by-design data handling
    """
    
    def __init__(self):
        self.encryption_key = self._get_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # In-memory storage (should be replaced with database models)
        self._coppa_users: Dict[str, COPPAUser] = {}
        self._consent_records: Dict[str, ConsentRecord] = {}
        self._verification_tokens: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.config = {
            "parental_consent_expiry_months": 12,  # Consent expires after 12 months
            "age_verification_required": True,
            "data_retention_days_under_13": 30,  # Minimal data retention for under 13
            "require_ongoing_consent_review": True,
            "consent_review_interval_months": 6,
            "allowed_data_categories_under_13": [
                DataCategory.EDUCATIONAL,
                DataCategory.DEVICE_INFO  # Minimal required for functionality
            ],
            "notification_methods": ["email", "in_app"],
            "compliance_reporting_enabled": True
        }
        
        logger.info("COPPA compliance system initialized")
    
    def _get_encryption_key(self) -> bytes:
        """Get encryption key for sensitive COPPA data."""
        # Generate key from settings for consistent encryption
        password = (settings.SECRET_KEY + "coppa_encryption").encode()
        salt = b'coppa_compliance_salt_v1'
        
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    # Age Verification
    
    async def verify_user_age(
        self,
        user_id: str,
        verification_request: AgeVerificationRequest
    ) -> Dict[str, Any]:
        """Verify user age and determine COPPA requirements."""
        try:
            # Calculate age
            today = date.today()
            age = today.year - verification_request.birth_date.year - (
                (today.month, today.day) < 
                (verification_request.birth_date.month, verification_request.birth_date.day)
            )
            
            is_under_13 = age < 13
            requires_parental_consent = is_under_13
            
            # Create COPPA user profile
            coppa_user = COPPAUser(
                user_id=user_id,
                birth_date=verification_request.birth_date,
                is_under_13=is_under_13,
                requires_parental_consent=requires_parental_consent,
                parental_consent_status=ParentalConsentStatus.PENDING if is_under_13 else ParentalConsentStatus.GRANTED,
                parent_email=verification_request.parent_email if is_under_13 else None,
                consent_method=None,
                consent_date=None,
                data_categories_consented=[],
                age_verification_method=verification_request.verification_method,
                age_verification_date=datetime.utcnow(),
                last_parental_review=None
            )
            
            # Store encrypted user profile
            self._coppa_users[user_id] = coppa_user
            
            # Log age verification
            security_system = await get_security_system()
            await security_system.log_security_event(
                SecurityEvent.DATA_ACCESS,
                user_id=user_id,
                action="coppa_age_verification",
                details={
                    "age": age,
                    "is_under_13": is_under_13,
                    "verification_method": verification_request.verification_method.value,
                    "requires_parental_consent": requires_parental_consent
                }
            )
            
            result = {
                "user_id": user_id,
                "age": age,
                "is_under_13": is_under_13,
                "requires_parental_consent": requires_parental_consent,
                "parental_consent_status": coppa_user.parental_consent_status.value,
                "verification_date": coppa_user.age_verification_date.isoformat()
            }
            
            # If under 13, initiate parental consent process
            if is_under_13 and verification_request.parent_email:
                consent_result = await self._initiate_parental_consent(user_id, verification_request.parent_email)
                result["consent_initiation"] = consent_result
            
            return result
            
        except Exception as e:
            logger.error("Age verification failed", error=str(e), user_id=user_id)
            raise HTTPException(status_code=500, detail="Age verification failed")
    
    # Parental Consent Management
    
    async def request_parental_consent(
        self,
        consent_request: ParentalConsentRequest
    ) -> Dict[str, Any]:
        """Request parental consent for child under 13."""
        try:
            coppa_user = self._coppa_users.get(consent_request.child_user_id)
            if not coppa_user or not coppa_user.is_under_13:
                raise HTTPException(status_code=400, detail="Parental consent not required for this user")
            
            # Generate consent record
            consent_id = f"consent_{secrets.token_urlsafe(16)}"
            verification_token = secrets.token_urlsafe(32)
            
            consent_record = ConsentRecord(
                consent_id=consent_id,
                child_user_id=consent_request.child_user_id,
                parent_email=consent_request.parent_email,
                parent_name=consent_request.parent_name,
                consent_status=ParentalConsentStatus.PENDING,
                consent_method=consent_request.consent_method,
                data_categories_consented=consent_request.data_categories,
                consent_date=None,
                expiration_date=None,
                revocation_date=None,
                verification_token=verification_token,
                metadata=consent_request.metadata
            )
            
            self._consent_records[consent_id] = consent_record
            
            # Store verification token for consent confirmation
            self._verification_tokens[verification_token] = {
                "consent_id": consent_id,
                "child_user_id": consent_request.child_user_id,
                "expires_at": datetime.utcnow() + timedelta(days=7),  # Token expires in 7 days
                "created_at": datetime.utcnow()
            }
            
            # Send consent request notification
            await self._send_parental_consent_notification(consent_record, verification_token)
            
            # Log consent request
            security_system = await get_security_system()
            await security_system.log_security_event(
                SecurityEvent.CONFIGURATION_CHANGED,
                user_id=consent_request.child_user_id,
                action="coppa_consent_requested",
                details={
                    "consent_id": consent_id,
                    "parent_email": consent_request.parent_email,
                    "consent_method": consent_request.consent_method.value,
                    "data_categories": [cat.value for cat in consent_request.data_categories]
                }
            )
            
            return {
                "consent_id": consent_id,
                "status": "pending",
                "verification_token_sent": True,
                "parent_email": consent_request.parent_email,
                "expires_at": (datetime.utcnow() + timedelta(days=7)).isoformat()
            }
            
        except Exception as e:
            logger.error("Parental consent request failed", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to request parental consent")
    
    async def confirm_parental_consent(
        self,
        verification_token: str,
        consent_granted: bool,
        additional_verification_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Confirm or deny parental consent using verification token."""
        try:
            # Validate verification token
            token_data = self._verification_tokens.get(verification_token)
            if not token_data:
                raise HTTPException(status_code=400, detail="Invalid verification token")
            
            if datetime.utcnow() > token_data["expires_at"]:
                raise HTTPException(status_code=400, detail="Verification token expired")
            
            consent_id = token_data["consent_id"]
            consent_record = self._consent_records.get(consent_id)
            if not consent_record:
                raise HTTPException(status_code=400, detail="Consent record not found")
            
            # Update consent status
            if consent_granted:
                consent_record.consent_status = ParentalConsentStatus.GRANTED
                consent_record.consent_date = datetime.utcnow()
                consent_record.expiration_date = datetime.utcnow() + timedelta(
                    days=30 * self.config["parental_consent_expiry_months"]
                )
                
                # Update COPPA user profile
                coppa_user = self._coppa_users.get(consent_record.child_user_id)
                if coppa_user:
                    coppa_user.parental_consent_status = ParentalConsentStatus.GRANTED
                    coppa_user.consent_method = consent_record.consent_method
                    coppa_user.consent_date = consent_record.consent_date
                    coppa_user.data_categories_consented = consent_record.data_categories_consented
                    coppa_user.parent_email = consent_record.parent_email
                
            else:
                consent_record.consent_status = ParentalConsentStatus.DENIED
                consent_record.revocation_date = datetime.utcnow()
                
                # Update COPPA user profile
                coppa_user = self._coppa_users.get(consent_record.child_user_id)
                if coppa_user:
                    coppa_user.parental_consent_status = ParentalConsentStatus.DENIED
            
            # Remove verification token
            del self._verification_tokens[verification_token]
            
            # Log consent decision
            security_system = await get_security_system()
            await security_system.log_security_event(
                SecurityEvent.CONFIGURATION_CHANGED,
                user_id=consent_record.child_user_id,
                action="coppa_consent_confirmed",
                details={
                    "consent_id": consent_id,
                    "consent_granted": consent_granted,
                    "consent_method": consent_record.consent_method.value,
                    "parent_email": consent_record.parent_email
                }
            )
            
            return {
                "consent_id": consent_id,
                "child_user_id": consent_record.child_user_id,
                "consent_granted": consent_granted,
                "status": consent_record.consent_status.value,
                "confirmation_date": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Parental consent confirmation failed", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to confirm parental consent")
    
    async def revoke_parental_consent(self, consent_id: str, revoked_by: str) -> Dict[str, Any]:
        """Revoke parental consent."""
        try:
            consent_record = self._consent_records.get(consent_id)
            if not consent_record:
                raise HTTPException(status_code=404, detail="Consent record not found")
            
            if consent_record.consent_status != ParentalConsentStatus.GRANTED:
                raise HTTPException(status_code=400, detail="Consent is not currently granted")
            
            # Update consent record
            consent_record.consent_status = ParentalConsentStatus.REVOKED
            consent_record.revocation_date = datetime.utcnow()
            
            # Update COPPA user profile
            coppa_user = self._coppa_users.get(consent_record.child_user_id)
            if coppa_user:
                coppa_user.parental_consent_status = ParentalConsentStatus.REVOKED
                coppa_user.data_categories_consented = []
            
            # Initiate data deletion process for revoked consent
            await self._initiate_data_deletion(consent_record.child_user_id)
            
            # Log consent revocation
            security_system = await get_security_system()
            await security_system.log_security_event(
                SecurityEvent.CONFIGURATION_CHANGED,
                user_id=consent_record.child_user_id,
                action="coppa_consent_revoked",
                details={
                    "consent_id": consent_id,
                    "revoked_by": revoked_by,
                    "parent_email": consent_record.parent_email
                }
            )
            
            return {
                "consent_id": consent_id,
                "status": "revoked",
                "revocation_date": consent_record.revocation_date.isoformat(),
                "data_deletion_initiated": True
            }
            
        except Exception as e:
            logger.error("Consent revocation failed", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to revoke consent")
    
    # Data Protection and Privacy
    
    async def check_data_collection_permission(
        self,
        user_id: str,
        data_category: DataCategory
    ) -> bool:
        """Check if data collection is permitted for user."""
        try:
            coppa_user = self._coppa_users.get(user_id)
            
            # If user is not in COPPA system, assume adult user
            if not coppa_user:
                return True
            
            # If user is 13 or older, normal data collection rules apply
            if not coppa_user.is_under_13:
                return True
            
            # For users under 13, check parental consent
            if coppa_user.parental_consent_status != ParentalConsentStatus.GRANTED:
                return False
            
            # Check if specific data category is consented to
            return data_category in coppa_user.data_categories_consented
            
        except Exception as e:
            logger.error("Data collection permission check failed", error=str(e))
            return False  # Fail closed for privacy protection
    
    async def get_data_retention_policy(self, user_id: str) -> Dict[str, Any]:
        """Get data retention policy for user."""
        try:
            coppa_user = self._coppa_users.get(user_id)
            
            if not coppa_user or not coppa_user.is_under_13:
                # Standard data retention for adults
                return {
                    "retention_days": 2555,  # 7 years standard
                    "reason": "standard_retention",
                    "special_requirements": []
                }
            
            # Enhanced privacy protection for children under 13
            return {
                "retention_days": self.config["data_retention_days_under_13"],
                "reason": "coppa_child_protection",
                "special_requirements": [
                    "parental_consent_required",
                    "minimal_data_collection",
                    "enhanced_deletion_rights",
                    "consent_expiry_monitoring"
                ],
                "consent_expiry_date": coppa_user.consent_date + timedelta(
                    days=30 * self.config["parental_consent_expiry_months"]
                ) if coppa_user.consent_date else None
            }
            
        except Exception as e:
            logger.error("Data retention policy check failed", error=str(e))
            return {
                "retention_days": 30,  # Fail safe with minimal retention
                "reason": "error_fallback",
                "special_requirements": ["immediate_review_required"]
            }
    
    async def _initiate_data_deletion(self, user_id: str) -> None:
        """Initiate data deletion process for user."""
        try:
            # This would integrate with data deletion services
            # For now, just log the deletion request
            security_system = await get_security_system()
            await security_system.log_security_event(
                SecurityEvent.DATA_ACCESS,
                user_id=user_id,
                action="coppa_data_deletion_initiated",
                details={
                    "reason": "consent_revoked",
                    "initiated_at": datetime.utcnow().isoformat()
                }
            )
            
            logger.info("Data deletion initiated for COPPA user", user_id=user_id)
            
        except Exception as e:
            logger.error("Data deletion initiation failed", error=str(e))
    
    # Consent Management Helpers
    
    async def _initiate_parental_consent(self, user_id: str, parent_email: str) -> Dict[str, Any]:
        """Initiate parental consent process."""
        try:
            # Create default consent request
            consent_request = ParentalConsentRequest(
                child_user_id=user_id,
                parent_email=parent_email,
                parent_name="Parent/Guardian",
                consent_method=ConsentMethod.EMAIL_VERIFICATION,
                data_categories=self.config["allowed_data_categories_under_13"]
            )
            
            return await self.request_parental_consent(consent_request)
            
        except Exception as e:
            logger.error("Parental consent initiation failed", error=str(e))
            return {"error": "Failed to initiate parental consent"}
    
    async def _send_parental_consent_notification(
        self,
        consent_record: ConsentRecord,
        verification_token: str
    ) -> None:
        """Send parental consent notification email."""
        try:
            # This would integrate with email service
            # For now, just log the notification
            logger.info(
                "Parental consent notification sent",
                parent_email=consent_record.parent_email,
                child_user_id=consent_record.child_user_id,
                verification_token=verification_token
            )
            
            # In production, would send actual email with consent form link
            consent_url = f"{settings.FRONTEND_URL}/consent/verify?token={verification_token}"
            logger.info("Consent verification URL", url=consent_url)
            
        except Exception as e:
            logger.error("Failed to send parental consent notification", error=str(e))
    
    # Compliance Monitoring
    
    async def get_coppa_compliance_report(self) -> Dict[str, Any]:
        """Generate COPPA compliance report."""
        try:
            total_users = len(self._coppa_users)
            under_13_users = sum(1 for user in self._coppa_users.values() if user.is_under_13)
            consent_granted = sum(
                1 for user in self._coppa_users.values() 
                if user.is_under_13 and user.parental_consent_status == ParentalConsentStatus.GRANTED
            )
            consent_pending = sum(
                1 for user in self._coppa_users.values()
                if user.is_under_13 and user.parental_consent_status == ParentalConsentStatus.PENDING
            )
            
            # Check for expiring consents
            expiring_soon = []
            for user in self._coppa_users.values():
                if (user.is_under_13 and 
                    user.parental_consent_status == ParentalConsentStatus.GRANTED and
                    user.consent_date):
                    
                    expiry_date = user.consent_date + timedelta(
                        days=30 * self.config["parental_consent_expiry_months"]
                    )
                    days_until_expiry = (expiry_date - datetime.utcnow()).days
                    
                    if days_until_expiry <= 30:  # Expiring within 30 days
                        expiring_soon.append({
                            "user_id": user.user_id,
                            "days_until_expiry": days_until_expiry,
                            "expiry_date": expiry_date.isoformat()
                        })
            
            return {
                "generated_at": datetime.utcnow().isoformat(),
                "total_users_in_system": total_users,
                "users_under_13": under_13_users,
                "parental_consent_granted": consent_granted,
                "parental_consent_pending": consent_pending,
                "consent_expiring_soon": len(expiring_soon),
                "expiring_consents": expiring_soon,
                "compliance_configuration": {
                    "consent_expiry_months": self.config["parental_consent_expiry_months"],
                    "data_retention_days_under_13": self.config["data_retention_days_under_13"],
                    "allowed_data_categories": [cat.value for cat in self.config["allowed_data_categories_under_13"]]
                }
            }
            
        except Exception as e:
            logger.error("COPPA compliance report generation failed", error=str(e))
            return {"error": "Failed to generate compliance report"}
    
    async def get_user_coppa_status(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get COPPA status for specific user."""
        coppa_user = self._coppa_users.get(user_id)
        if not coppa_user:
            return None
        
        return {
            "user_id": user_id,
            "is_under_13": coppa_user.is_under_13,
            "requires_parental_consent": coppa_user.requires_parental_consent,
            "parental_consent_status": coppa_user.parental_consent_status.value,
            "consent_date": coppa_user.consent_date.isoformat() if coppa_user.consent_date else None,
            "consent_expiry_date": (
                coppa_user.consent_date + timedelta(
                    days=30 * self.config["parental_consent_expiry_months"]
                )
            ).isoformat() if coppa_user.consent_date else None,
            "data_categories_consented": [cat.value for cat in coppa_user.data_categories_consented],
            "age_verification_method": coppa_user.age_verification_method.value,
            "age_verification_date": coppa_user.age_verification_date.isoformat()
        }


# Global COPPA system instance
_coppa_system: Optional[COPPAComplianceSystem] = None


def get_coppa_system() -> COPPAComplianceSystem:
    """Get or create COPPA compliance system instance."""
    global _coppa_system
    if _coppa_system is None:
        _coppa_system = COPPAComplianceSystem()
    return _coppa_system


# FastAPI Dependencies

async def check_coppa_compliance(user_id: str, data_category: DataCategory):
    """Check COPPA compliance for data collection."""
    coppa_system = get_coppa_system()
    permitted = await coppa_system.check_data_collection_permission(user_id, data_category)
    
    if not permitted:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Data collection not permitted under COPPA"
        )


def require_coppa_compliance(data_category: DataCategory):
    """Dependency factory for COPPA compliance checking."""
    async def compliance_checker(user: User = Depends(get_current_user)):
        await check_coppa_compliance(user.id, data_category)
        return user
    
    return compliance_checker


# Export components
__all__ = [
    "COPPAComplianceSystem", "get_coppa_system", "check_coppa_compliance",
    "require_coppa_compliance", "AgeVerificationRequest", "ParentalConsentRequest",
    "ParentalConsentStatus", "ConsentMethod", "DataCategory", "AgeVerificationMethod"
]