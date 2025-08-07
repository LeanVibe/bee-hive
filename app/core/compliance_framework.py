"""
SOC2/GDPR Compliance Framework for LeanVibe Agent Hive 2.0
Implements comprehensive compliance controls and audit capabilities
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path

import aioredis
from cryptography.fernet import Fernet
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from fastapi import HTTPException, status
from pydantic import BaseModel, Field, validator

from app.models.user import User
from app.models.agent import Agent
from app.core.database import get_session

logger = logging.getLogger(__name__)

class ComplianceFramework(str, Enum):
    """Supported compliance frameworks"""
    SOC2_TYPE_II = "soc2_type_ii"
    GDPR = "gdpr"
    CCPA = "ccpa"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"

class DataClassification(str, Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class ProcessingLawfulBasis(str, Enum):
    """GDPR lawful basis for processing"""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"

@dataclass
class ComplianceEvent:
    """Compliance audit event"""
    event_id: str
    timestamp: datetime
    event_type: str
    user_id: Optional[str]
    agent_id: Optional[str]
    data_classification: DataClassification
    processing_basis: Optional[ProcessingLawfulBasis]
    description: str
    metadata: Dict[str, Any]
    frameworks: List[ComplianceFramework]
    retention_period_days: int = 2555  # 7 years default
    
class DataRetentionPolicy:
    """Data retention policy implementation"""
    
    def __init__(self):
        self.policies = {
            DataClassification.PUBLIC: 365,  # 1 year
            DataClassification.INTERNAL: 1095,  # 3 years
            DataClassification.CONFIDENTIAL: 2555,  # 7 years
            DataClassification.RESTRICTED: 2555  # 7 years
        }
        
    def get_retention_period(self, classification: DataClassification) -> int:
        """Get retention period in days for data classification"""
        return self.policies.get(classification, 2555)
    
    def is_retention_expired(self, created_date: datetime, classification: DataClassification) -> bool:
        """Check if data retention period has expired"""
        retention_days = self.get_retention_period(classification)
        expiry_date = created_date + timedelta(days=retention_days)
        return datetime.utcnow() > expiry_date

class GDPRComplianceManager:
    """GDPR compliance management"""
    
    def __init__(self, redis_client: aioredis.Redis, db_session_factory):
        self.redis = redis_client
        self.db_session_factory = db_session_factory
        self.retention_policy = DataRetentionPolicy()
        
    async def log_data_processing(
        self,
        user_id: str,
        processing_purpose: str,
        data_categories: List[str],
        lawful_basis: ProcessingLawfulBasis,
        retention_period_days: Optional[int] = None
    ) -> str:
        """Log GDPR data processing activity"""
        
        event_id = f"gdpr_{int(time.time())}_{hashlib.sha256(f'{user_id}{processing_purpose}'.encode()).hexdigest()[:8]}"
        
        processing_record = {
            "event_id": event_id,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "processing_purpose": processing_purpose,
            "data_categories": data_categories,
            "lawful_basis": lawful_basis.value,
            "retention_period_days": retention_period_days or 2555,
            "status": "active"
        }
        
        # Store in Redis for fast access
        await self.redis.hset(f"gdpr_processing:{user_id}", event_id, json.dumps(processing_record))
        await self.redis.expire(f"gdpr_processing:{user_id}", 86400 * 7)  # 7 day cache
        
        # Store in audit log
        await self._store_compliance_event(
            ComplianceEvent(
                event_id=event_id,
                timestamp=datetime.utcnow(),
                event_type="data_processing",
                user_id=user_id,
                agent_id=None,
                data_classification=DataClassification.CONFIDENTIAL,
                processing_basis=lawful_basis,
                description=f"Data processing: {processing_purpose}",
                metadata=processing_record,
                frameworks=[ComplianceFramework.GDPR]
            )
        )
        
        return event_id
    
    async def handle_data_subject_request(
        self,
        user_id: str,
        request_type: str,  # access, rectification, erasure, portability, restriction
        request_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle GDPR data subject requests"""
        
        if request_type == "access":
            return await self._handle_access_request(user_id)
        elif request_type == "erasure":
            return await self._handle_erasure_request(user_id, request_details)
        elif request_type == "portability":
            return await self._handle_portability_request(user_id)
        elif request_type == "rectification":
            return await self._handle_rectification_request(user_id, request_details)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported request type: {request_type}"
            )
    
    async def _handle_access_request(self, user_id: str) -> Dict[str, Any]:
        """Handle GDPR Article 15 - Right of Access"""
        
        async with self.db_session_factory() as session:
            # Gather all user data
            user_data = {}
            
            # User profile data
            user_query = select(User).where(User.id == user_id)
            user = await session.execute(user_query)
            user_record = user.scalar_one_or_none()
            
            if user_record:
                user_data["profile"] = {
                    "id": str(user_record.id),
                    "username": user_record.username,
                    "email": user_record.email,
                    "created_at": user_record.created_at.isoformat(),
                    "last_login": user_record.last_login.isoformat() if user_record.last_login else None
                }
            
            # Processing activities
            processing_key = f"gdpr_processing:{user_id}"
            processing_data = await self.redis.hgetall(processing_key)
            user_data["processing_activities"] = [
                json.loads(data.decode()) for data in processing_data.values()
            ]
            
            # Consent records
            consent_key = f"gdpr_consent:{user_id}"
            consent_data = await self.redis.hgetall(consent_key)
            user_data["consents"] = [
                json.loads(data.decode()) for data in consent_data.values()
            ]
            
            # Log the access request
            await self._store_compliance_event(
                ComplianceEvent(
                    event_id=f"access_{int(time.time())}_{user_id}",
                    timestamp=datetime.utcnow(),
                    event_type="data_subject_access",
                    user_id=user_id,
                    agent_id=None,
                    data_classification=DataClassification.CONFIDENTIAL,
                    processing_basis=ProcessingLawfulBasis.LEGAL_OBLIGATION,
                    description="GDPR Article 15 - Data Subject Access Request",
                    metadata={"request_fulfilled": True, "data_categories": list(user_data.keys())},
                    frameworks=[ComplianceFramework.GDPR]
                )
            )
            
            return {
                "request_type": "access",
                "user_id": user_id,
                "data": user_data,
                "processed_at": datetime.utcnow().isoformat(),
                "retention_info": self._get_retention_info(user_data)
            }
    
    async def _handle_erasure_request(self, user_id: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """Handle GDPR Article 17 - Right to Erasure"""
        
        async with self.db_session_factory() as session:
            # Check if erasure is legally possible
            legal_holds = await self._check_legal_holds(user_id)
            
            if legal_holds:
                return {
                    "request_type": "erasure",
                    "user_id": user_id,
                    "status": "rejected",
                    "reason": "Data subject to legal hold",
                    "legal_holds": legal_holds,
                    "processed_at": datetime.utcnow().isoformat()
                }
            
            # Perform erasure
            erasure_results = []
            
            # Anonymize user profile
            user_query = select(User).where(User.id == user_id)
            user = await session.execute(user_query)
            user_record = user.scalar_one_or_none()
            
            if user_record:
                # Anonymize instead of delete to maintain referential integrity
                user_record.username = f"anonymized_{hashlib.sha256(user_id.encode()).hexdigest()[:8]}"
                user_record.email = f"anonymized_{hashlib.sha256(user_id.encode()).hexdigest()[:8]}@deleted.local"
                user_record.first_name = "DELETED"
                user_record.last_name = "DELETED"
                await session.commit()
                erasure_results.append("user_profile_anonymized")
            
            # Delete Redis data
            await self.redis.delete(f"gdpr_processing:{user_id}")
            await self.redis.delete(f"gdpr_consent:{user_id}")
            await self.redis.delete(f"user_session:{user_id}")
            erasure_results.append("redis_data_deleted")
            
            # Log the erasure
            await self._store_compliance_event(
                ComplianceEvent(
                    event_id=f"erasure_{int(time.time())}_{user_id}",
                    timestamp=datetime.utcnow(),
                    event_type="data_subject_erasure",
                    user_id=user_id,
                    agent_id=None,
                    data_classification=DataClassification.CONFIDENTIAL,
                    processing_basis=ProcessingLawfulBasis.LEGAL_OBLIGATION,
                    description="GDPR Article 17 - Right to Erasure",
                    metadata={"erasure_results": erasure_results},
                    frameworks=[ComplianceFramework.GDPR]
                )
            )
            
            return {
                "request_type": "erasure",
                "user_id": user_id,
                "status": "completed",
                "erasure_results": erasure_results,
                "processed_at": datetime.utcnow().isoformat()
            }
    
    async def _check_legal_holds(self, user_id: str) -> List[str]:
        """Check if user data is subject to legal holds"""
        legal_holds = []
        
        # Check for active legal proceedings
        legal_hold_key = f"legal_hold:{user_id}"
        holds = await self.redis.smembers(legal_hold_key)
        
        for hold in holds:
            legal_holds.append(hold.decode())
        
        return legal_holds
    
    def _get_retention_info(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get retention information for user data"""
        retention_info = {}
        
        for category, data in user_data.items():
            if isinstance(data, dict) and "created_at" in data:
                created_date = datetime.fromisoformat(data["created_at"])
                classification = DataClassification.CONFIDENTIAL  # Default
                retention_days = self.retention_policy.get_retention_period(classification)
                expiry_date = created_date + timedelta(days=retention_days)
                
                retention_info[category] = {
                    "retention_period_days": retention_days,
                    "expires_at": expiry_date.isoformat(),
                    "days_remaining": (expiry_date - datetime.utcnow()).days
                }
        
        return retention_info

class SOC2ComplianceManager:
    """SOC2 Type II compliance management"""
    
    def __init__(self, redis_client: aioredis.Redis, db_session_factory):
        self.redis = redis_client
        self.db_session_factory = db_session_factory
        
    async def log_security_control(
        self,
        control_id: str,
        control_description: str,
        evidence: Dict[str, Any],
        operator_id: Optional[str] = None
    ) -> str:
        """Log SOC2 security control execution"""
        
        event_id = f"soc2_{int(time.time())}_{control_id}"
        
        control_record = {
            "event_id": event_id,
            "timestamp": datetime.utcnow().isoformat(),
            "control_id": control_id,
            "control_description": control_description,
            "evidence": evidence,
            "operator_id": operator_id,
            "status": "executed"
        }
        
        # Store control execution
        await self.redis.lpush("soc2_controls", json.dumps(control_record))
        await self.redis.ltrim("soc2_controls", 0, 9999)  # Keep last 10,000
        
        # Log compliance event
        await self._store_compliance_event(
            ComplianceEvent(
                event_id=event_id,
                timestamp=datetime.utcnow(),
                event_type="security_control",
                user_id=operator_id,
                agent_id=None,
                data_classification=DataClassification.INTERNAL,
                processing_basis=None,
                description=f"SOC2 Control: {control_description}",
                metadata=control_record,
                frameworks=[ComplianceFramework.SOC2_TYPE_II]
            )
        )
        
        return event_id
    
    async def monitor_trust_services_criteria(self) -> Dict[str, Any]:
        """Monitor SOC2 Trust Services Criteria"""
        
        criteria_status = {}
        
        # Security (CC1-CC8)
        criteria_status["security"] = await self._assess_security_criteria()
        
        # Availability (A1)
        criteria_status["availability"] = await self._assess_availability_criteria()
        
        # Processing Integrity (PI1)
        criteria_status["processing_integrity"] = await self._assess_processing_integrity_criteria()
        
        # Confidentiality (C1)
        criteria_status["confidentiality"] = await self._assess_confidentiality_criteria()
        
        # Privacy (P1-P8)
        criteria_status["privacy"] = await self._assess_privacy_criteria()
        
        return criteria_status
    
    async def _assess_security_criteria(self) -> Dict[str, Any]:
        """Assess SOC2 Security criteria"""
        
        # Check access controls
        access_control_score = await self._check_access_controls()
        
        # Check system monitoring
        monitoring_score = await self._check_system_monitoring()
        
        # Check incident response
        incident_response_score = await self._check_incident_response()
        
        return {
            "overall_score": (access_control_score + monitoring_score + incident_response_score) / 3,
            "access_controls": access_control_score,
            "system_monitoring": monitoring_score,
            "incident_response": incident_response_score,
            "assessment_time": datetime.utcnow().isoformat()
        }
    
    async def _check_access_controls(self) -> float:
        """Check access control implementation"""
        controls_checked = 0
        controls_passed = 0
        
        # Check 1: Multi-factor authentication enabled
        mfa_enabled = await self.redis.get("security_control:mfa_enabled")
        controls_checked += 1
        if mfa_enabled == b"true":
            controls_passed += 1
        
        # Check 2: Role-based access controls
        rbac_enabled = await self.redis.get("security_control:rbac_enabled")
        controls_checked += 1
        if rbac_enabled == b"true":
            controls_passed += 1
        
        # Check 3: Regular access reviews
        last_access_review = await self.redis.get("security_control:last_access_review")
        controls_checked += 1
        if last_access_review:
            review_date = datetime.fromisoformat(last_access_review.decode())
            if (datetime.utcnow() - review_date).days <= 90:
                controls_passed += 1
        
        return (controls_passed / controls_checked) * 100 if controls_checked > 0 else 0

class ComplianceAuditManager:
    """Unified compliance audit management"""
    
    def __init__(self, redis_client: aioredis.Redis, db_session_factory):
        self.redis = redis_client
        self.db_session_factory = db_session_factory
        self.gdpr_manager = GDPRComplianceManager(redis_client, db_session_factory)
        self.soc2_manager = SOC2ComplianceManager(redis_client, db_session_factory)
        
    async def generate_compliance_report(
        self,
        framework: ComplianceFramework,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        
        report = {
            "framework": framework.value,
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "generated_at": datetime.utcnow().isoformat(),
            "summary": {},
            "events": [],
            "metrics": {},
            "recommendations": []
        }
        
        # Get compliance events for period
        events = await self._get_compliance_events(framework, start_date, end_date)
        report["events"] = events
        
        # Calculate metrics
        report["metrics"] = await self._calculate_compliance_metrics(framework, events)
        
        # Generate summary
        report["summary"] = await self._generate_compliance_summary(framework, events)
        
        # Generate recommendations
        report["recommendations"] = await self._generate_recommendations(framework, report["metrics"])
        
        return report
    
    async def _get_compliance_events(
        self,
        framework: ComplianceFramework,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Retrieve compliance events for reporting period"""
        
        events = []
        
        # Get events from Redis
        event_keys = await self.redis.keys(f"compliance_event:*:{framework.value}")
        
        for key in event_keys:
            event_data = await self.redis.hgetall(key)
            if event_data:
                event = {}
                for field, value in event_data.items():
                    try:
                        event[field.decode()] = json.loads(value.decode())
                    except json.JSONDecodeError:
                        event[field.decode()] = value.decode()
                
                # Filter by date range
                event_time = datetime.fromisoformat(event["timestamp"])
                if start_date <= event_time <= end_date:
                    events.append(event)
        
        return sorted(events, key=lambda x: x["timestamp"])
    
    async def _store_compliance_event(self, event: ComplianceEvent):
        """Store compliance event for audit trail"""
        
        event_key = f"compliance_event:{event.event_id}"
        event_data = {
            "event_id": event.event_id,
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type,
            "user_id": event.user_id,
            "agent_id": event.agent_id,
            "data_classification": event.data_classification.value,
            "processing_basis": event.processing_basis.value if event.processing_basis else None,
            "description": event.description,
            "metadata": json.dumps(event.metadata),
            "frameworks": json.dumps([f.value for f in event.frameworks]),
            "retention_period_days": event.retention_period_days
        }
        
        # Store in Redis
        await self.redis.hset(event_key, mapping=event_data)
        
        # Set expiration based on retention period
        await self.redis.expire(event_key, event.retention_period_days * 86400)
        
        # Index by framework for quick retrieval
        for framework in event.frameworks:
            framework_key = f"compliance_events:{framework.value}"
            await self.redis.zadd(framework_key, {event.event_id: time.time()})
            await self.redis.expire(framework_key, event.retention_period_days * 86400)
    
    async def run_automated_compliance_checks(self) -> Dict[str, Any]:
        """Run automated compliance validation checks"""
        
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {},
            "overall_status": "UNKNOWN",
            "issues_found": [],
            "recommendations": []
        }
        
        # GDPR checks
        gdpr_results = await self._run_gdpr_checks()
        results["checks"]["gdpr"] = gdpr_results
        
        # SOC2 checks
        soc2_results = await self._run_soc2_checks()
        results["checks"]["soc2"] = soc2_results
        
        # Determine overall status
        all_checks = [gdpr_results["status"], soc2_results["status"]]
        if all(status == "COMPLIANT" for status in all_checks):
            results["overall_status"] = "COMPLIANT"
        elif any(status == "NON_COMPLIANT" for status in all_checks):
            results["overall_status"] = "NON_COMPLIANT"
        else:
            results["overall_status"] = "NEEDS_REVIEW"
        
        # Aggregate issues and recommendations
        for check_result in results["checks"].values():
            results["issues_found"].extend(check_result.get("issues", []))
            results["recommendations"].extend(check_result.get("recommendations", []))
        
        return results
    
    async def _run_gdpr_checks(self) -> Dict[str, Any]:
        """Run GDPR compliance checks"""
        
        issues = []
        recommendations = []
        
        # Check data retention compliance
        expired_data = await self._check_data_retention()
        if expired_data:
            issues.append(f"Found {len(expired_data)} records past retention period")
            recommendations.append("Implement automated data purging process")
        
        # Check consent management
        consent_issues = await self._check_consent_management()
        issues.extend(consent_issues)
        
        # Check data subject request handling
        pending_requests = await self._check_pending_data_subject_requests()
        if pending_requests > 0:
            issues.append(f"{pending_requests} pending data subject requests")
            recommendations.append("Review and process pending requests within 30 days")
        
        status = "NON_COMPLIANT" if issues else "COMPLIANT"
        
        return {
            "framework": "GDPR",
            "status": status,
            "issues": issues,
            "recommendations": recommendations,
            "last_checked": datetime.utcnow().isoformat()
        }
    
    async def _run_soc2_checks(self) -> Dict[str, Any]:
        """Run SOC2 compliance checks"""
        
        issues = []
        recommendations = []
        
        # Check security controls
        security_score = await self.soc2_manager._check_access_controls()
        if security_score < 90:
            issues.append(f"Security controls score below threshold: {security_score}%")
            recommendations.append("Review and strengthen access controls")
        
        # Check monitoring and logging
        log_retention = await self._check_log_retention()
        if not log_retention["compliant"]:
            issues.append("Security logs retention not compliant")
            recommendations.append("Ensure security logs are retained for minimum 1 year")
        
        status = "NON_COMPLIANT" if issues else "COMPLIANT"
        
        return {
            "framework": "SOC2",
            "status": status,
            "issues": issues,
            "recommendations": recommendations,
            "last_checked": datetime.utcnow().isoformat()
        }

# FastAPI integration for compliance endpoints
class ComplianceAPI:
    """API endpoints for compliance management"""
    
    def __init__(self, compliance_manager: ComplianceAuditManager):
        self.compliance_manager = compliance_manager
    
    async def handle_data_subject_request(
        self,
        user_id: str,
        request_type: str,
        request_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle GDPR data subject requests via API"""
        
        return await self.compliance_manager.gdpr_manager.handle_data_subject_request(
            user_id, request_type, request_details
        )
    
    async def generate_compliance_report(
        self,
        framework: str,
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """Generate compliance report via API"""
        
        framework_enum = ComplianceFramework(framework)
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        return await self.compliance_manager.generate_compliance_report(
            framework_enum, start_dt, end_dt
        )

async def create_compliance_framework(
    redis_url: str,
    db_session_factory
) -> ComplianceAuditManager:
    """Factory function to create compliance framework"""
    
    redis_client = aioredis.from_url(redis_url)
    return ComplianceAuditManager(redis_client, db_session_factory)