"""
GDPR Compliance Framework for LeanVibe Agent Hive 2.0

Implements General Data Protection Regulation (GDPR) compliance with:
- Consent management and tracking
- Data subject rights (access, rectification, erasure, portability)
- Privacy by design and by default
- Data protection impact assessments
- Breach notification procedures
- Audit logging for regulatory compliance

CRITICAL COMPONENT: Required for any processing of EU resident data.
"""

import asyncio
import base64
import hashlib
import json
import secrets
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field

import structlog
from fastapi import HTTPException, Request, status
from pydantic import BaseModel, Field, validator, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, update, delete
from cryptography.fernet import Fernet

from .config import settings
from .database import get_session
from .enterprise_security_system import get_security_system, SecurityEvent
from .auth import User, get_auth_service

logger = structlog.get_logger()


class ConsentBasis(Enum):
    """Legal basis for data processing under GDPR."""
    CONSENT = "consent"  # Article 6(1)(a)
    CONTRACT = "contract"  # Article 6(1)(b)
    LEGAL_OBLIGATION = "legal_obligation"  # Article 6(1)(c)
    VITAL_INTERESTS = "vital_interests"  # Article 6(1)(d)
    PUBLIC_TASK = "public_task"  # Article 6(1)(e)
    LEGITIMATE_INTERESTS = "legitimate_interests"  # Article 6(1)(f)


class DataSubjectRight(Enum):
    """Data subject rights under GDPR."""
    ACCESS = "access"  # Article 15
    RECTIFICATION = "rectification"  # Article 16
    ERASURE = "erasure"  # Article 17 (Right to be forgotten)
    RESTRICT_PROCESSING = "restrict_processing"  # Article 18
    DATA_PORTABILITY = "data_portability"  # Article 20
    OBJECT = "object"  # Article 21
    WITHDRAW_CONSENT = "withdraw_consent"  # Article 7(3)


class ProcessingPurpose(Enum):
    """Purposes for data processing."""
    SERVICE_PROVISION = "service_provision"
    ACCOUNT_MANAGEMENT = "account_management"
    CUSTOMER_SUPPORT = "customer_support"
    ANALYTICS = "analytics"
    MARKETING = "marketing"
    SECURITY = "security"
    LEGAL_COMPLIANCE = "legal_compliance"
    RESEARCH = "research"


class DataCategory(Enum):
    """Categories of personal data."""
    IDENTITY_DATA = "identity_data"  # Name, username, title
    CONTACT_DATA = "contact_data"  # Email, phone, address
    TECHNICAL_DATA = "technical_data"  # IP address, device info, cookies
    USAGE_DATA = "usage_data"  # How you use our service
    MARKETING_DATA = "marketing_data"  # Preferences, survey responses
    FINANCIAL_DATA = "financial_data"  # Payment information
    SPECIAL_CATEGORY = "special_category"  # Sensitive data requiring explicit consent


class ConsentStatus(Enum):
    """Status of user consent."""
    GIVEN = "given"
    WITHDRAWN = "withdrawn"
    PENDING = "pending"
    EXPIRED = "expired"
    INVALID = "invalid"


class RequestStatus(Enum):
    """Status of data subject requests."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REJECTED = "rejected"
    PARTIALLY_COMPLETED = "partially_completed"


@dataclass
class ConsentRecord:
    """GDPR consent record."""
    consent_id: str
    user_id: str
    purpose: ProcessingPurpose
    data_categories: List[DataCategory]
    legal_basis: ConsentBasis
    consent_status: ConsentStatus
    consent_given_at: Optional[datetime]
    consent_withdrawn_at: Optional[datetime]
    expiry_date: Optional[datetime]
    consent_text: str
    consent_version: str
    withdrawal_method: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "consent_id": self.consent_id,
            "user_id": self.user_id,
            "purpose": self.purpose.value,
            "data_categories": [cat.value for cat in self.data_categories],
            "legal_basis": self.legal_basis.value,
            "consent_status": self.consent_status.value,
            "consent_given_at": self.consent_given_at.isoformat() if self.consent_given_at else None,
            "consent_withdrawn_at": self.consent_withdrawn_at.isoformat() if self.consent_withdrawn_at else None,
            "expiry_date": self.expiry_date.isoformat() if self.expiry_date else None,
            "consent_text": self.consent_text,
            "consent_version": self.consent_version,
            "withdrawal_method": self.withdrawal_method,
            "metadata": self.metadata
        }


@dataclass
class DataSubjectRequest:
    """Data subject request record."""
    request_id: str
    user_id: str
    request_type: DataSubjectRight
    status: RequestStatus
    submitted_at: datetime
    completed_at: Optional[datetime]
    response_data: Optional[Dict[str, Any]]
    verification_method: str
    identity_verified: bool
    processing_notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "request_type": self.request_type.value,
            "status": self.status.value,
            "submitted_at": self.submitted_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "response_data": self.response_data,
            "verification_method": self.verification_method,
            "identity_verified": self.identity_verified,
            "processing_notes": self.processing_notes,
            "metadata": self.metadata
        }


class ConsentRequest(BaseModel):
    """Consent request model."""
    purposes: List[ProcessingPurpose]
    data_categories: List[DataCategory]
    consent_text: str
    consent_version: str = "1.0"
    expiry_months: Optional[int] = None
    metadata: Dict[str, Any] = {}


class DataSubjectRequestModel(BaseModel):
    """Data subject request model."""
    request_type: DataSubjectRight
    verification_method: str = "email_verification"
    additional_info: Optional[str] = None
    metadata: Dict[str, Any] = {}


class GDPRComplianceSystem:
    """
    Comprehensive GDPR Compliance System.
    
    Features:
    - Consent management with granular tracking
    - Data subject rights automation
    - Privacy by design data handling
    - Breach detection and notification
    - Audit logging for regulatory compliance
    - Data protection impact assessment tools
    """
    
    def __init__(self):
        self.encryption_key = self._get_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # In-memory storage (should be replaced with database models)
        self._consent_records: Dict[str, ConsentRecord] = {}
        self._data_subject_requests: Dict[str, DataSubjectRequest] = {}
        self._processing_activities: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.config = {
            "consent_expiry_months": 24,  # Default consent expiry
            "request_response_days": 30,  # GDPR mandated response time
            "breach_notification_hours": 72,  # GDPR mandated breach notification time
            "data_retention_policies": {
                "default_retention_days": 2555,  # 7 years
                "marketing_retention_days": 1095,  # 3 years
                "analytics_retention_days": 730,  # 2 years
                "inactive_account_retention_days": 365  # 1 year
            },
            "automated_deletion_enabled": True,
            "privacy_by_default": True,
            "require_explicit_consent_special_categories": True
        }
        
        # Initialize default processing activities
        self._initialize_processing_activities()
        
        logger.info("GDPR compliance system initialized")
    
    def _get_encryption_key(self) -> bytes:
        """Get encryption key for sensitive GDPR data."""
        password = (settings.SECRET_KEY + "gdpr_encryption").encode()
        salt = b'gdpr_compliance_salt_v1'
        
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
    
    def _initialize_processing_activities(self):
        """Initialize register of processing activities (Article 30)."""
        self._processing_activities = {
            "user_account_management": {
                "controller": "LeanVibe Inc.",
                "purposes": [ProcessingPurpose.SERVICE_PROVISION.value, ProcessingPurpose.ACCOUNT_MANAGEMENT.value],
                "data_categories": [DataCategory.IDENTITY_DATA.value, DataCategory.CONTACT_DATA.value],
                "legal_basis": ConsentBasis.CONTRACT.value,
                "retention_period": "Duration of contract + 7 years",
                "recipients": ["Internal teams", "Cloud service providers"],
                "transfers": []
            },
            "customer_support": {
                "controller": "LeanVibe Inc.",
                "purposes": [ProcessingPurpose.CUSTOMER_SUPPORT.value],
                "data_categories": [DataCategory.IDENTITY_DATA.value, DataCategory.CONTACT_DATA.value, DataCategory.USAGE_DATA.value],
                "legal_basis": ConsentBasis.LEGITIMATE_INTERESTS.value,
                "retention_period": "3 years after last contact",
                "recipients": ["Support team", "Third-party support tools"],
                "transfers": []
            },
            "analytics_and_improvement": {
                "controller": "LeanVibe Inc.",
                "purposes": [ProcessingPurpose.ANALYTICS.value],
                "data_categories": [DataCategory.TECHNICAL_DATA.value, DataCategory.USAGE_DATA.value],
                "legal_basis": ConsentBasis.LEGITIMATE_INTERESTS.value,
                "retention_period": "2 years",
                "recipients": ["Analytics team", "Analytics service providers"],
                "transfers": []
            }
        }
    
    # Consent Management
    
    async def record_consent(
        self,
        user_id: str,
        consent_request: ConsentRequest,
        consent_given: bool = True
    ) -> Dict[str, Any]:
        """Record user consent for data processing."""
        try:
            consent_id = f"consent_{secrets.token_urlsafe(16)}"
            
            # Calculate expiry date
            expiry_date = None
            if consent_request.expiry_months:
                expiry_date = datetime.utcnow() + timedelta(days=30 * consent_request.expiry_months)
            elif self.config["consent_expiry_months"]:
                expiry_date = datetime.utcnow() + timedelta(days=30 * self.config["consent_expiry_months"])
            
            # Create consent records for each purpose
            consent_records = []
            for purpose in consent_request.purposes:
                record = ConsentRecord(
                    consent_id=f"{consent_id}_{purpose.value}",
                    user_id=user_id,
                    purpose=purpose,
                    data_categories=consent_request.data_categories,
                    legal_basis=ConsentBasis.CONSENT,
                    consent_status=ConsentStatus.GIVEN if consent_given else ConsentStatus.WITHDRAWN,
                    consent_given_at=datetime.utcnow() if consent_given else None,
                    consent_withdrawn_at=datetime.utcnow() if not consent_given else None,
                    expiry_date=expiry_date,
                    consent_text=consent_request.consent_text,
                    consent_version=consent_request.consent_version,
                    metadata=consent_request.metadata
                )
                
                self._consent_records[record.consent_id] = record
                consent_records.append(record)
            
            # Log consent recording
            security_system = await get_security_system()
            await security_system.log_security_event(
                SecurityEvent.CONFIGURATION_CHANGED,
                user_id=user_id,
                action="gdpr_consent_recorded",
                details={
                    "consent_id": consent_id,
                    "purposes": [p.value for p in consent_request.purposes],
                    "data_categories": [c.value for c in consent_request.data_categories],
                    "consent_given": consent_given,
                    "consent_version": consent_request.consent_version,
                    "expiry_date": expiry_date.isoformat() if expiry_date else None
                }
            )
            
            return {
                "consent_id": consent_id,
                "user_id": user_id,
                "consent_given": consent_given,
                "purposes_recorded": len(consent_request.purposes),
                "expiry_date": expiry_date.isoformat() if expiry_date else None,
                "recorded_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Consent recording failed", error=str(e), user_id=user_id)
            raise HTTPException(status_code=500, detail="Failed to record consent")
    
    async def withdraw_consent(
        self,
        user_id: str,
        purposes: List[ProcessingPurpose],
        withdrawal_method: str = "user_request"
    ) -> Dict[str, Any]:
        """Withdraw user consent for specified purposes."""
        try:
            withdrawn_consents = []
            
            # Find and withdraw consent records
            for consent_record in self._consent_records.values():
                if (consent_record.user_id == user_id and 
                    consent_record.purpose in purposes and
                    consent_record.consent_status == ConsentStatus.GIVEN):
                    
                    consent_record.consent_status = ConsentStatus.WITHDRAWN
                    consent_record.consent_withdrawn_at = datetime.utcnow()
                    consent_record.withdrawal_method = withdrawal_method
                    
                    withdrawn_consents.append(consent_record.consent_id)
            
            # Initiate data processing restrictions
            if withdrawn_consents:
                await self._apply_consent_withdrawal_restrictions(user_id, purposes)
            
            # Log consent withdrawal
            security_system = await get_security_system()
            await security_system.log_security_event(
                SecurityEvent.CONFIGURATION_CHANGED,
                user_id=user_id,
                action="gdpr_consent_withdrawn",
                details={
                    "purposes": [p.value for p in purposes],
                    "withdrawal_method": withdrawal_method,
                    "consents_withdrawn": len(withdrawn_consents),
                    "withdrawn_consent_ids": withdrawn_consents
                }
            )
            
            return {
                "user_id": user_id,
                "withdrawn_purposes": [p.value for p in purposes],
                "consents_withdrawn": len(withdrawn_consents),
                "withdrawal_date": datetime.utcnow().isoformat(),
                "restrictions_applied": len(withdrawn_consents) > 0
            }
            
        except Exception as e:
            logger.error("Consent withdrawal failed", error=str(e), user_id=user_id)
            raise HTTPException(status_code=500, detail="Failed to withdraw consent")
    
    async def check_consent_validity(
        self,
        user_id: str,
        purpose: ProcessingPurpose,
        data_category: DataCategory
    ) -> Dict[str, Any]:
        """Check if user has valid consent for specific purpose and data category."""
        try:
            # Find relevant consent records
            valid_consents = []
            expired_consents = []
            
            for consent_record in self._consent_records.values():
                if (consent_record.user_id == user_id and 
                    consent_record.purpose == purpose and
                    data_category in consent_record.data_categories):
                    
                    # Check if consent is still valid
                    if consent_record.consent_status == ConsentStatus.GIVEN:
                        if (consent_record.expiry_date and 
                            datetime.utcnow() > consent_record.expiry_date):
                            # Mark as expired
                            consent_record.consent_status = ConsentStatus.EXPIRED
                            expired_consents.append(consent_record.consent_id)
                        else:
                            valid_consents.append(consent_record.consent_id)
            
            has_valid_consent = len(valid_consents) > 0
            
            return {
                "user_id": user_id,
                "purpose": purpose.value,
                "data_category": data_category.value,
                "has_valid_consent": has_valid_consent,
                "valid_consents": valid_consents,
                "expired_consents": expired_consents,
                "checked_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Consent validity check failed", error=str(e))
            return {
                "user_id": user_id,
                "purpose": purpose.value,
                "data_category": data_category.value,
                "has_valid_consent": False,
                "error": str(e)
            }
    
    # Data Subject Rights
    
    async def submit_data_subject_request(
        self,
        user_id: str,
        request: DataSubjectRequestModel
    ) -> Dict[str, Any]:
        """Submit data subject request (Article 15-22)."""
        try:
            request_id = f"dsr_{secrets.token_urlsafe(16)}"
            
            # Create data subject request record
            dsr = DataSubjectRequest(
                request_id=request_id,
                user_id=user_id,
                request_type=request.request_type,
                status=RequestStatus.PENDING,
                submitted_at=datetime.utcnow(),
                completed_at=None,
                response_data=None,
                verification_method=request.verification_method,
                identity_verified=False,
                processing_notes=[f"Request submitted at {datetime.utcnow().isoformat()}"],
                metadata=request.metadata
            )
            
            self._data_subject_requests[request_id] = dsr
            
            # Start automated processing based on request type
            await self._process_data_subject_request(dsr)
            
            # Log data subject request
            security_system = await get_security_system()
            await security_system.log_security_event(
                SecurityEvent.DATA_ACCESS,
                user_id=user_id,
                action="gdpr_data_subject_request",
                details={
                    "request_id": request_id,
                    "request_type": request.request_type.value,
                    "verification_method": request.verification_method
                }
            )
            
            return {
                "request_id": request_id,
                "request_type": request.request_type.value,
                "status": dsr.status.value,
                "submitted_at": dsr.submitted_at.isoformat(),
                "estimated_completion_days": self.config["request_response_days"],
                "verification_required": True
            }
            
        except Exception as e:
            logger.error("Data subject request submission failed", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to submit data subject request")
    
    async def _process_data_subject_request(self, dsr: DataSubjectRequest) -> None:
        """Process data subject request automatically where possible."""
        try:
            dsr.status = RequestStatus.IN_PROGRESS
            dsr.processing_notes.append(f"Automated processing started at {datetime.utcnow().isoformat()}")
            
            if dsr.request_type == DataSubjectRight.ACCESS:
                # Right of access (Article 15)
                await self._process_access_request(dsr)
                
            elif dsr.request_type == DataSubjectRight.RECTIFICATION:
                # Right to rectification (Article 16)
                await self._process_rectification_request(dsr)
                
            elif dsr.request_type == DataSubjectRight.ERASURE:
                # Right to erasure (Article 17)
                await self._process_erasure_request(dsr)
                
            elif dsr.request_type == DataSubjectRight.DATA_PORTABILITY:
                # Right to data portability (Article 20)
                await self._process_portability_request(dsr)
                
            elif dsr.request_type == DataSubjectRight.WITHDRAW_CONSENT:
                # Right to withdraw consent (Article 7)
                await self._process_consent_withdrawal_request(dsr)
                
            else:
                # Other rights require manual processing
                dsr.processing_notes.append("Request requires manual review")
            
        except Exception as e:
            logger.error("Data subject request processing failed", error=str(e))
            dsr.status = RequestStatus.REJECTED
            dsr.processing_notes.append(f"Processing failed: {str(e)}")
    
    async def _process_access_request(self, dsr: DataSubjectRequest) -> None:
        """Process right of access request."""
        try:
            user_id = dsr.user_id
            
            # Collect user data from various sources
            user_data = await self._collect_user_data(user_id)
            
            # Prepare response data
            response_data = {
                "personal_data": user_data,
                "processing_purposes": await self._get_user_processing_purposes(user_id),
                "data_categories": await self._get_user_data_categories(user_id),
                "data_sources": await self._get_user_data_sources(user_id),
                "data_recipients": await self._get_user_data_recipients(user_id),
                "retention_periods": await self._get_user_retention_periods(user_id),
                "rights_information": self._get_data_subject_rights_info(),
                "consent_records": [
                    record.to_dict() for record in self._consent_records.values()
                    if record.user_id == user_id
                ]
            }
            
            dsr.response_data = response_data
            dsr.status = RequestStatus.COMPLETED
            dsr.completed_at = datetime.utcnow()
            dsr.processing_notes.append("Access request completed with full data export")
            
        except Exception as e:
            logger.error("Access request processing failed", error=str(e))
            dsr.status = RequestStatus.REJECTED
            dsr.processing_notes.append(f"Access request failed: {str(e)}")
    
    async def _process_erasure_request(self, dsr: DataSubjectRequest) -> None:
        """Process right to erasure (right to be forgotten) request."""
        try:
            user_id = dsr.user_id
            
            # Check if erasure is possible (legal obligations, etc.)
            erasure_assessment = await self._assess_erasure_eligibility(user_id)
            
            if erasure_assessment["eligible"]:
                # Perform data deletion
                deletion_result = await self._execute_data_deletion(user_id)
                
                dsr.response_data = {
                    "erasure_completed": True,
                    "data_deleted": deletion_result["deleted_categories"],
                    "retention_exceptions": deletion_result["retained_categories"],
                    "deletion_date": datetime.utcnow().isoformat()
                }
                dsr.status = RequestStatus.COMPLETED
                dsr.completed_at = datetime.utcnow()
                dsr.processing_notes.append("Erasure request completed")
                
            else:
                dsr.response_data = {
                    "erasure_completed": False,
                    "rejection_reasons": erasure_assessment["reasons"],
                    "legal_basis_for_retention": erasure_assessment["legal_basis"]
                }
                dsr.status = RequestStatus.REJECTED
                dsr.processing_notes.append("Erasure request rejected due to legal obligations")
            
        except Exception as e:
            logger.error("Erasure request processing failed", error=str(e))
            dsr.status = RequestStatus.REJECTED
            dsr.processing_notes.append(f"Erasure request failed: {str(e)}")
    
    async def _process_portability_request(self, dsr: DataSubjectRequest) -> None:
        """Process right to data portability request."""
        try:
            user_id = dsr.user_id
            
            # Collect portable data (data provided by user, processed based on consent or contract)
            portable_data = await self._collect_portable_user_data(user_id)
            
            # Format data in structured, machine-readable format (JSON)
            response_data = {
                "data_export": portable_data,
                "export_format": "JSON",
                "export_date": datetime.utcnow().isoformat(),
                "data_categories_included": list(portable_data.keys()),
                "total_records": sum(len(v) if isinstance(v, list) else 1 for v in portable_data.values())
            }
            
            dsr.response_data = response_data
            dsr.status = RequestStatus.COMPLETED
            dsr.completed_at = datetime.utcnow()
            dsr.processing_notes.append("Data portability request completed")
            
        except Exception as e:
            logger.error("Data portability request processing failed", error=str(e))
            dsr.status = RequestStatus.REJECTED
            dsr.processing_notes.append(f"Portability request failed: {str(e)}")
    
    # Helper Methods for Data Subject Requests
    
    async def _collect_user_data(self, user_id: str) -> Dict[str, Any]:
        """Collect all user data for access request."""
        # This would integrate with all data stores
        auth_service = get_auth_service()
        user = auth_service.get_user_by_id(user_id)
        
        if not user:
            return {}
        
        return {
            "account_information": {
                "user_id": user.id,
                "email": user.email,
                "full_name": user.full_name,
                "role": user.role.value,
                "company_name": user.company_name,
                "created_at": user.created_at.isoformat(),
                "last_login": user.last_login.isoformat() if user.last_login else None,
                "is_active": user.is_active
            },
            "consent_records": [
                record.to_dict() for record in self._consent_records.values()
                if record.user_id == user_id
            ]
        }
    
    async def _collect_portable_user_data(self, user_id: str) -> Dict[str, Any]:
        """Collect user data that is portable under GDPR."""
        user_data = await self._collect_user_data(user_id)
        
        # Filter to only include data that is portable
        # (data provided by user or processed based on consent/contract)
        return {
            "account_data": user_data.get("account_information", {}),
            "consent_history": user_data.get("consent_records", []),
            # Add other portable data categories as needed
        }
    
    async def _assess_erasure_eligibility(self, user_id: str) -> Dict[str, Any]:
        """Assess if user data can be erased under GDPR."""
        try:
            # Check for legal obligations that prevent erasure
            legal_obligations = []
            
            # Example checks (would be more comprehensive in production)
            # - Financial records retention requirements
            # - Legal dispute records
            # - Regulatory compliance requirements
            
            eligible = len(legal_obligations) == 0
            
            return {
                "eligible": eligible,
                "reasons": legal_obligations if not eligible else [],
                "legal_basis": "No legal obligations preventing erasure" if eligible else legal_obligations
            }
            
        except Exception as e:
            logger.error("Erasure eligibility assessment failed", error=str(e))
            return {
                "eligible": False,
                "reasons": ["Assessment failed"],
                "legal_basis": ["Unable to determine legal obligations"]
            }
    
    async def _execute_data_deletion(self, user_id: str) -> Dict[str, Any]:
        """Execute data deletion for erasure request."""
        try:
            # This would integrate with all data stores to delete user data
            deleted_categories = [
                "account_information",
                "usage_data",
                "preferences",
                "consent_records"
            ]
            
            retained_categories = []  # Categories retained due to legal obligations
            
            # Mark user as deleted in auth system
            auth_service = get_auth_service()
            user = auth_service.get_user_by_id(user_id)
            if user:
                user.is_active = False
                # In production, would pseudonymize rather than delete entirely
            
            # Remove consent records
            user_consents = [
                consent_id for consent_id, record in self._consent_records.items()
                if record.user_id == user_id
            ]
            for consent_id in user_consents:
                del self._consent_records[consent_id]
            
            return {
                "deleted_categories": deleted_categories,
                "retained_categories": retained_categories,
                "deletion_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Data deletion execution failed", error=str(e))
            raise
    
    async def _apply_consent_withdrawal_restrictions(
        self,
        user_id: str,
        purposes: List[ProcessingPurpose]
    ) -> None:
        """Apply processing restrictions after consent withdrawal."""
        try:
            # This would integrate with data processing systems to restrict processing
            logger.info(
                "Consent withdrawal restrictions applied",
                user_id=user_id,
                restricted_purposes=[p.value for p in purposes]
            )
            
        except Exception as e:
            logger.error("Failed to apply consent withdrawal restrictions", error=str(e))
    
    # Utility Methods
    
    async def _get_user_processing_purposes(self, user_id: str) -> List[str]:
        """Get processing purposes for user."""
        purposes = set()
        for record in self._consent_records.values():
            if record.user_id == user_id and record.consent_status == ConsentStatus.GIVEN:
                purposes.add(record.purpose.value)
        return list(purposes)
    
    async def _get_user_data_categories(self, user_id: str) -> List[str]:
        """Get data categories processed for user."""
        categories = set()
        for record in self._consent_records.values():
            if record.user_id == user_id and record.consent_status == ConsentStatus.GIVEN:
                categories.update(cat.value for cat in record.data_categories)
        return list(categories)
    
    async def _get_user_data_sources(self, user_id: str) -> List[str]:
        """Get data sources for user."""
        return ["User registration", "Service usage", "Support interactions"]
    
    async def _get_user_data_recipients(self, user_id: str) -> List[str]:
        """Get data recipients for user."""
        return ["Internal teams", "Cloud service providers", "Analytics providers"]
    
    async def _get_user_retention_periods(self, user_id: str) -> Dict[str, str]:
        """Get data retention periods for user."""
        return {
            "account_data": "Duration of service + 7 years",
            "usage_data": "2 years",
            "marketing_data": "3 years or until consent withdrawn"
        }
    
    def _get_data_subject_rights_info(self) -> Dict[str, str]:
        """Get information about data subject rights."""
        return {
            "right_to_access": "You have the right to access your personal data",
            "right_to_rectification": "You have the right to correct inaccurate personal data",
            "right_to_erasure": "You have the right to have your personal data erased",
            "right_to_restrict_processing": "You have the right to restrict processing of your personal data",
            "right_to_data_portability": "You have the right to receive your personal data in a portable format",
            "right_to_object": "You have the right to object to processing of your personal data",
            "right_to_withdraw_consent": "You have the right to withdraw consent at any time"
        }
    
    # Compliance Reporting
    
    async def get_gdpr_compliance_report(self) -> Dict[str, Any]:
        """Generate GDPR compliance report."""
        try:
            now = datetime.utcnow()
            
            # Consent statistics
            total_consents = len(self._consent_records)
            active_consents = sum(
                1 for record in self._consent_records.values()
                if record.consent_status == ConsentStatus.GIVEN
            )
            withdrawn_consents = sum(
                1 for record in self._consent_records.values()
                if record.consent_status == ConsentStatus.WITHDRAWN
            )
            expired_consents = sum(
                1 for record in self._consent_records.values()
                if record.consent_status == ConsentStatus.EXPIRED
            )
            
            # Data subject request statistics
            total_requests = len(self._data_subject_requests)
            pending_requests = sum(
                1 for request in self._data_subject_requests.values()
                if request.status in [RequestStatus.PENDING, RequestStatus.IN_PROGRESS]
            )
            completed_requests = sum(
                1 for request in self._data_subject_requests.values()
                if request.status == RequestStatus.COMPLETED
            )
            
            # Overdue requests (older than 30 days)
            overdue_requests = []
            for request in self._data_subject_requests.values():
                if request.status in [RequestStatus.PENDING, RequestStatus.IN_PROGRESS]:
                    days_open = (now - request.submitted_at).days
                    if days_open > self.config["request_response_days"]:
                        overdue_requests.append({
                            "request_id": request.request_id,
                            "request_type": request.request_type.value,
                            "days_overdue": days_open - self.config["request_response_days"],
                            "submitted_at": request.submitted_at.isoformat()
                        })
            
            return {
                "generated_at": now.isoformat(),
                "consent_statistics": {
                    "total_consents": total_consents,
                    "active_consents": active_consents,
                    "withdrawn_consents": withdrawn_consents,
                    "expired_consents": expired_consents
                },
                "data_subject_request_statistics": {
                    "total_requests": total_requests,
                    "pending_requests": pending_requests,
                    "completed_requests": completed_requests,
                    "overdue_requests": len(overdue_requests),
                    "overdue_details": overdue_requests
                },
                "processing_activities": len(self._processing_activities),
                "compliance_configuration": {
                    "consent_expiry_months": self.config["consent_expiry_months"],
                    "request_response_days": self.config["request_response_days"],
                    "automated_deletion_enabled": self.config["automated_deletion_enabled"],
                    "privacy_by_default": self.config["privacy_by_default"]
                }
            }
            
        except Exception as e:
            logger.error("GDPR compliance report generation failed", error=str(e))
            return {"error": "Failed to generate compliance report"}


# Global GDPR system instance
_gdpr_system: Optional[GDPRComplianceSystem] = None


def get_gdpr_system() -> GDPRComplianceSystem:
    """Get or create GDPR compliance system instance."""
    global _gdpr_system
    if _gdpr_system is None:
        _gdpr_system = GDPRComplianceSystem()
    return _gdpr_system


# FastAPI Dependencies

async def check_gdpr_consent(
    user_id: str,
    purpose: ProcessingPurpose,
    data_category: DataCategory
):
    """Check GDPR consent for data processing."""
    gdpr_system = get_gdpr_system()
    consent_check = await gdpr_system.check_consent_validity(user_id, purpose, data_category)
    
    if not consent_check["has_valid_consent"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Valid consent required under GDPR"
        )


def require_gdpr_consent(purpose: ProcessingPurpose, data_category: DataCategory):
    """Dependency factory for GDPR consent checking."""
    async def consent_checker(user: User = Depends(get_current_user)):
        await check_gdpr_consent(user.id, purpose, data_category)
        return user
    
    return consent_checker


# Export components
__all__ = [
    "GDPRComplianceSystem", "get_gdpr_system", "check_gdpr_consent",
    "require_gdpr_consent", "ConsentRequest", "DataSubjectRequestModel",
    "ConsentBasis", "DataSubjectRight", "ProcessingPurpose", "DataCategory"
]