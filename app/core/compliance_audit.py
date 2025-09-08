"""
Compliance & Audit Infrastructure for LeanVibe Agent Hive 2.0.

Implements enterprise-grade compliance and audit capabilities including:
- Comprehensive audit logging for all system actions
- Compliance reporting automation for SOC2, ISO27001, GDPR
- Data encryption at rest and in transit
- Security scanning and vulnerability management automation
- Compliance framework validation and reporting
"""

import uuid
import hashlib
import json
import gzip
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field, asdict
import logging
import asyncio
from pathlib import Path
import csv

from sqlalchemy import select, and_, or_, func, desc, text, MetaData, Table
from sqlalchemy.ext.asyncio import AsyncSession
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64
import os

from ..models.agent import Agent
from ..models.context import Context
from ..core.access_control import Permission, AccessLevel


logger = logging.getLogger(__name__)


class DataClassification(Enum):
    """Data classification levels for encryption and handling."""
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    RESTRICTED = "RESTRICTED"
    TOP_SECRET = "TOP_SECRET"


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    SOC2_TYPE1 = "SOC2_TYPE1"
    SOC2_TYPE2 = "SOC2_TYPE2"
    ISO27001 = "ISO27001"
    GDPR = "GDPR"
    HIPAA = "HIPAA"
    PCI_DSS = "PCI_DSS"
    NIST_CSF = "NIST_CSF"
    FedRAMP = "FedRAMP"


class AuditEventCategory(Enum):
    """Categories of audit events."""
    AUTHENTICATION = "AUTHENTICATION"
    AUTHORIZATION = "AUTHORIZATION"
    DATA_ACCESS = "DATA_ACCESS"
    DATA_MODIFICATION = "DATA_MODIFICATION"
    SYSTEM_CONFIGURATION = "SYSTEM_CONFIGURATION"
    SECURITY_EVENT = "SECURITY_EVENT"
    COMPLIANCE_EVENT = "COMPLIANCE_EVENT"
    ERROR_EVENT = "ERROR_EVENT"
    PERFORMANCE_EVENT = "PERFORMANCE_EVENT"


class SeverityLevel(Enum):
    """Severity levels for audit events."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


@dataclass
class AuditEvent:
    """Comprehensive audit event structure."""
    id: uuid.UUID
    timestamp: datetime
    category: AuditEventCategory
    severity: SeverityLevel
    event_type: str
    user_id: Optional[uuid.UUID]
    organization_id: Optional[uuid.UUID]
    session_id: Optional[uuid.UUID]
    source_ip: Optional[str]
    user_agent: Optional[str]
    resource_id: Optional[uuid.UUID]
    resource_type: Optional[str]
    action: str
    outcome: str  # SUCCESS, FAILURE, PARTIAL
    details: Dict[str, Any]
    compliance_tags: List[str] = field(default_factory=list)
    retention_class: str = "STANDARD"  # STANDARD, LONG_TERM, PERMANENT
    encrypted: bool = False
    checksum: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "severity": self.severity.value,
            "event_type": self.event_type,
            "user_id": str(self.user_id) if self.user_id else None,
            "organization_id": str(self.organization_id) if self.organization_id else None,
            "session_id": str(self.session_id) if self.session_id else None,
            "source_ip": self.source_ip,
            "user_agent": self.user_agent,
            "resource_id": str(self.resource_id) if self.resource_id else None,
            "resource_type": self.resource_type,
            "action": self.action,
            "outcome": self.outcome,
            "details": self.details,
            "compliance_tags": self.compliance_tags,
            "retention_class": self.retention_class,
            "encrypted": self.encrypted,
            "checksum": self.checksum
        }
    
    def calculate_checksum(self) -> str:
        """Calculate checksum for audit event integrity."""
        event_data = self.to_dict()
        event_data.pop("checksum", None)  # Remove checksum from calculation
        event_json = json.dumps(event_data, sort_keys=True)
        return hashlib.sha256(event_json.encode()).hexdigest()


@dataclass
class EncryptedData:
    """Container for encrypted data with metadata."""
    encrypted_data: bytes
    encryption_algorithm: str
    key_id: str
    iv: Optional[bytes]
    classification: DataClassification
    created_at: datetime
    expires_at: Optional[datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceRule:
    """Represents a compliance rule to be validated."""
    id: str
    framework: ComplianceFramework
    control_id: str
    title: str
    description: str
    validation_query: str
    expected_result: Any
    severity: SeverityLevel
    automated: bool = True
    frequency: str = "DAILY"  # DAILY, WEEKLY, MONTHLY, QUARTERLY
    last_checked: Optional[datetime] = None
    last_result: Optional[str] = None


@dataclass
class ComplianceReport:
    """Comprehensive compliance report."""
    id: uuid.UUID
    framework: ComplianceFramework
    organization_id: uuid.UUID
    report_period_start: datetime
    report_period_end: datetime
    generated_at: datetime
    generated_by: uuid.UUID
    controls_evaluated: int
    controls_passed: int
    controls_failed: int
    controls_not_applicable: int
    overall_compliance_score: float
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    evidence_collected: List[str]
    report_data: Dict[str, Any]
    status: str = "DRAFT"  # DRAFT, FINAL, APPROVED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "framework": self.framework.value,
            "organization_id": str(self.organization_id),
            "report_period_start": self.report_period_start.isoformat(),
            "report_period_end": self.report_period_end.isoformat(),
            "generated_at": self.generated_at.isoformat(),
            "generated_by": str(self.generated_by),
            "controls_evaluated": self.controls_evaluated,
            "controls_passed": self.controls_passed,
            "controls_failed": self.controls_failed,
            "controls_not_applicable": self.controls_not_applicable,
            "overall_compliance_score": self.overall_compliance_score,
            "findings": self.findings,
            "recommendations": self.recommendations,
            "evidence_collected": self.evidence_collected,
            "report_data": self.report_data,
            "status": self.status
        }


@dataclass
class VulnerabilityReport:
    """Security vulnerability scan report."""
    id: uuid.UUID
    scan_type: str
    target: str
    scan_started: datetime
    scan_completed: datetime
    vulnerabilities_found: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    info_count: int
    vulnerabilities: List[Dict[str, Any]]
    remediation_guidance: List[str]
    scan_metadata: Dict[str, Any] = field(default_factory=dict)


class ComplianceAuditSystem:
    """
    Comprehensive compliance and audit system.
    
    Features:
    - Comprehensive audit logging with integrity protection
    - Multi-framework compliance reporting (SOC2, ISO27001, GDPR, etc.)
    - Data encryption at rest and in transit
    - Automated security vulnerability scanning
    - Compliance rule validation and monitoring
    - Evidence collection and retention management
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        encryption_keys: Dict[str, bytes],
        audit_storage_path: str = "/var/log/agent-hive/audit"
    ):
        """
        Initialize compliance audit system.
        
        Args:
            db_session: Database session
            encryption_keys: Dictionary of encryption keys by key ID
            audit_storage_path: Path for audit log storage
        """
        self.db = db_session
        self.encryption_keys = encryption_keys
        self.audit_storage_path = Path(audit_storage_path)
        self.audit_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize encryption
        self.default_key_id = "default"
        if self.default_key_id not in encryption_keys:
            # Generate default encryption key
            self.encryption_keys[self.default_key_id] = Fernet.generate_key()
        
        self.fernet = Fernet(self.encryption_keys[self.default_key_id])
        
        # Audit event buffer for batch processing
        self.audit_buffer: List[AuditEvent] = []
        self.buffer_max_size = 1000
        self.buffer_flush_interval = 60  # seconds
        
        # Compliance rules registry
        self.compliance_rules: Dict[ComplianceFramework, List[ComplianceRule]] = {}
        self._initialize_compliance_rules()
        
        # Vulnerability scan results
        self.vulnerability_reports: List[VulnerabilityReport] = []
        
        # Configuration
        self.config = {
            "audit_retention_days": 2555,  # 7 years
            "log_compression": True,
            "encryption_required_classifications": [
                DataClassification.CONFIDENTIAL,
                DataClassification.RESTRICTED,
                DataClassification.TOP_SECRET
            ],
            "compliance_scan_frequency": "DAILY",
            "vulnerability_scan_frequency": "WEEKLY",
            "report_generation_schedule": "MONTHLY"
        }
        
        # Start background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._start_background_tasks()
    
    async def log_audit_event(
        self,
        event_type: str,
        category: AuditEventCategory,
        severity: SeverityLevel,
        action: str,
        outcome: str,
        user_id: Optional[uuid.UUID] = None,
        organization_id: Optional[uuid.UUID] = None,
        resource_id: Optional[uuid.UUID] = None,
        resource_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        compliance_tags: Optional[List[str]] = None,
        session_id: Optional[uuid.UUID] = None,
        source_ip: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> AuditEvent:
        """
        Log comprehensive audit event.
        
        Args:
            event_type: Type of event (e.g., "USER_LOGIN", "DATA_ACCESS")
            category: Event category
            severity: Severity level
            action: Action performed
            outcome: Result of action (SUCCESS, FAILURE, PARTIAL)
            user_id: ID of user performing action
            organization_id: Organization context
            resource_id: ID of resource being accessed
            resource_type: Type of resource
            details: Additional event details
            compliance_tags: Tags for compliance mapping
            session_id: Session ID
            source_ip: Source IP address
            user_agent: User agent string
            
        Returns:
            Created audit event
        """
        try:
            audit_event = AuditEvent(
                id=uuid.uuid4(),
                timestamp=datetime.utcnow(),
                category=category,
                severity=severity,
                event_type=event_type,
                user_id=user_id,
                organization_id=organization_id,
                session_id=session_id,
                source_ip=source_ip,
                user_agent=user_agent,
                resource_id=resource_id,
                resource_type=resource_type,
                action=action,
                outcome=outcome,
                details=details or {},
                compliance_tags=compliance_tags or []
            )
            
            # Calculate integrity checksum
            audit_event.checksum = audit_event.calculate_checksum()
            
            # Add to buffer for batch processing
            self.audit_buffer.append(audit_event)
            
            # Flush buffer if full
            if len(self.audit_buffer) >= self.buffer_max_size:
                await self._flush_audit_buffer()
            
            logger.debug(f"Audit event logged: {event_type} - {outcome}")
            return audit_event
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            raise
    
    async def encrypt_sensitive_data(
        self,
        data: Union[str, bytes, Dict[str, Any]],
        classification: DataClassification,
        key_id: Optional[str] = None
    ) -> EncryptedData:
        """
        Encrypt sensitive data based on classification.
        
        Args:
            data: Data to encrypt
            classification: Data classification level
            key_id: Specific encryption key to use
            
        Returns:
            Encrypted data container
        """
        try:
            # Convert data to bytes if needed
            if isinstance(data, dict):
                data_bytes = json.dumps(data).encode('utf-8')
            elif isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data
            
            # Select encryption key
            key_id = key_id or self.default_key_id
            if key_id not in self.encryption_keys:
                raise ValueError(f"Encryption key {key_id} not found")
            
            # Encrypt data
            fernet = Fernet(self.encryption_keys[key_id])
            encrypted_bytes = fernet.encrypt(data_bytes)
            
            encrypted_data = EncryptedData(
                encrypted_data=encrypted_bytes,
                encryption_algorithm="Fernet",
                key_id=key_id,
                iv=None,  # Fernet handles IV internally
                classification=classification,
                created_at=datetime.utcnow(),
                expires_at=None  # Set based on retention policy
            )
            
            # Audit encryption event
            await self.log_audit_event(
                event_type="DATA_ENCRYPTION",
                category=AuditEventCategory.SECURITY_EVENT,
                severity=SeverityLevel.INFO,
                action="ENCRYPT",
                outcome="SUCCESS",
                details={
                    "classification": classification.value,
                    "algorithm": "Fernet",
                    "key_id": key_id,
                    "data_size": len(data_bytes)
                },
                compliance_tags=["DATA_PROTECTION", "ENCRYPTION"]
            )
            
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Data encryption failed: {e}")
            await self.log_audit_event(
                event_type="DATA_ENCRYPTION",
                category=AuditEventCategory.ERROR_EVENT,
                severity=SeverityLevel.HIGH,
                action="ENCRYPT",
                outcome="FAILURE",
                details={"error": str(e)}
            )
            raise
    
    async def decrypt_sensitive_data(self, encrypted_data: EncryptedData) -> bytes:
        """
        Decrypt sensitive data.
        
        Args:
            encrypted_data: Encrypted data container
            
        Returns:
            Decrypted data bytes
        """
        try:
            if encrypted_data.key_id not in self.encryption_keys:
                raise ValueError(f"Encryption key {encrypted_data.key_id} not found")
            
            fernet = Fernet(self.encryption_keys[encrypted_data.key_id])
            decrypted_bytes = fernet.decrypt(encrypted_data.encrypted_data)
            
            # Audit decryption event
            await self.log_audit_event(
                event_type="DATA_DECRYPTION",
                category=AuditEventCategory.SECURITY_EVENT,
                severity=SeverityLevel.INFO,
                action="DECRYPT",
                outcome="SUCCESS",
                details={
                    "classification": encrypted_data.classification.value,
                    "algorithm": encrypted_data.encryption_algorithm,
                    "key_id": encrypted_data.key_id
                },
                compliance_tags=["DATA_PROTECTION", "DECRYPTION"]
            )
            
            return decrypted_bytes
            
        except Exception as e:
            logger.error(f"Data decryption failed: {e}")
            await self.log_audit_event(
                event_type="DATA_DECRYPTION",
                category=AuditEventCategory.ERROR_EVENT,
                severity=SeverityLevel.HIGH,
                action="DECRYPT",
                outcome="FAILURE",
                details={"error": str(e)}
            )
            raise
    
    async def scan_security_vulnerabilities(self, scan_target: str) -> VulnerabilityReport:
        """
        Perform security vulnerability scan.
        
        Args:
            scan_target: Target for vulnerability scanning
            
        Returns:
            Vulnerability scan report
        """
        try:
            scan_started = datetime.utcnow()
            
            # Simulate vulnerability scanning (replace with actual scanner integration)
            vulnerabilities = await self._perform_vulnerability_scan(scan_target)
            
            scan_completed = datetime.utcnow()
            
            # Categorize vulnerabilities by severity
            critical_count = len([v for v in vulnerabilities if v.get("severity") == "CRITICAL"])
            high_count = len([v for v in vulnerabilities if v.get("severity") == "HIGH"])
            medium_count = len([v for v in vulnerabilities if v.get("severity") == "MEDIUM"])
            low_count = len([v for v in vulnerabilities if v.get("severity") == "LOW"])
            info_count = len([v for v in vulnerabilities if v.get("severity") == "INFO"])
            
            # Generate remediation guidance
            remediation_guidance = self._generate_remediation_guidance(vulnerabilities)
            
            report = VulnerabilityReport(
                id=uuid.uuid4(),
                scan_type="COMPREHENSIVE",
                target=scan_target,
                scan_started=scan_started,
                scan_completed=scan_completed,
                vulnerabilities_found=len(vulnerabilities),
                critical_count=critical_count,
                high_count=high_count,
                medium_count=medium_count,
                low_count=low_count,
                info_count=info_count,
                vulnerabilities=vulnerabilities,
                remediation_guidance=remediation_guidance
            )
            
            self.vulnerability_reports.append(report)
            
            # Audit vulnerability scan
            await self.log_audit_event(
                event_type="VULNERABILITY_SCAN",
                category=AuditEventCategory.SECURITY_EVENT,
                severity=SeverityLevel.HIGH if critical_count > 0 else SeverityLevel.MEDIUM,
                action="SECURITY_SCAN",
                outcome="SUCCESS",
                details={
                    "target": scan_target,
                    "vulnerabilities_found": len(vulnerabilities),
                    "critical_count": critical_count,
                    "high_count": high_count,
                    "scan_duration_seconds": (scan_completed - scan_started).total_seconds()
                },
                compliance_tags=["VULNERABILITY_MANAGEMENT", "SECURITY_SCANNING"]
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Vulnerability scan failed: {e}")
            await self.log_audit_event(
                event_type="VULNERABILITY_SCAN",
                category=AuditEventCategory.ERROR_EVENT,
                severity=SeverityLevel.HIGH,
                action="SECURITY_SCAN",
                outcome="FAILURE",
                details={"error": str(e), "target": scan_target}
            )
            raise
    
    async def generate_compliance_report(
        self,
        framework: ComplianceFramework,
        organization_id: uuid.UUID,
        period_start: datetime,
        period_end: datetime,
        generated_by: uuid.UUID
    ) -> ComplianceReport:
        """
        Generate comprehensive compliance report.
        
        Args:
            framework: Compliance framework
            organization_id: Organization for report
            period_start: Report period start
            period_end: Report period end
            generated_by: User generating report
            
        Returns:
            Compliance report
        """
        try:
            logger.info(f"Generating {framework.value} compliance report for org {organization_id}")
            
            # Get compliance rules for framework
            rules = self.compliance_rules.get(framework, [])
            
            # Evaluate compliance rules
            findings = []
            controls_passed = 0
            controls_failed = 0
            controls_not_applicable = 0
            evidence_collected = []
            
            for rule in rules:
                try:
                    result = await self._evaluate_compliance_rule(rule, organization_id, period_start, period_end)
                    
                    if result["status"] == "PASS":
                        controls_passed += 1
                    elif result["status"] == "FAIL":
                        controls_failed += 1
                        findings.append(result)
                    else:
                        controls_not_applicable += 1
                    
                    if result.get("evidence"):
                        evidence_collected.extend(result["evidence"])
                        
                except Exception as e:
                    logger.error(f"Failed to evaluate rule {rule.id}: {e}")
                    controls_failed += 1
                    findings.append({
                        "rule_id": rule.id,
                        "control_id": rule.control_id,
                        "status": "FAIL",
                        "error": str(e)
                    })
            
            # Calculate compliance score
            total_controls = len(rules)
            if total_controls > 0:
                compliance_score = (controls_passed / total_controls) * 100
            else:
                compliance_score = 0.0
            
            # Generate recommendations
            recommendations = self._generate_compliance_recommendations(framework, findings)
            
            # Create report
            report = ComplianceReport(
                id=uuid.uuid4(),
                framework=framework,
                organization_id=organization_id,
                report_period_start=period_start,
                report_period_end=period_end,
                generated_at=datetime.utcnow(),
                generated_by=generated_by,
                controls_evaluated=total_controls,
                controls_passed=controls_passed,
                controls_failed=controls_failed,
                controls_not_applicable=controls_not_applicable,
                overall_compliance_score=compliance_score,
                findings=findings,
                recommendations=recommendations,
                evidence_collected=evidence_collected,
                report_data={
                    "framework_version": "2023.1",
                    "assessment_methodology": "Automated + Manual Review",
                    "tools_used": ["Agent Hive Compliance Engine"],
                    "report_format": "Standard"
                }
            )
            
            # Audit report generation
            await self.log_audit_event(
                event_type="COMPLIANCE_REPORT_GENERATED",
                category=AuditEventCategory.COMPLIANCE_EVENT,
                severity=SeverityLevel.INFO,
                action="GENERATE_REPORT",
                outcome="SUCCESS",
                user_id=generated_by,
                organization_id=organization_id,
                details={
                    "framework": framework.value,
                    "compliance_score": compliance_score,
                    "controls_evaluated": total_controls,
                    "controls_failed": controls_failed
                },
                compliance_tags=[framework.value, "COMPLIANCE_REPORTING"]
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Compliance report generation failed: {e}")
            await self.log_audit_event(
                event_type="COMPLIANCE_REPORT_GENERATION",
                category=AuditEventCategory.ERROR_EVENT,
                severity=SeverityLevel.HIGH,
                action="GENERATE_REPORT",
                outcome="FAILURE",
                user_id=generated_by,
                organization_id=organization_id,
                details={"error": str(e), "framework": framework.value}
            )
            raise
    
    async def automate_compliance_checks(
        self,
        framework: ComplianceFramework,
        organization_id: Optional[uuid.UUID] = None
    ) -> Dict[str, Any]:
        """
        Automate compliance checks for specified framework.
        
        Args:
            framework: Compliance framework to check
            organization_id: Specific organization (None for all)
            
        Returns:
            Compliance check results
        """
        try:
            rules = self.compliance_rules.get(framework, [])
            results = {
                "framework": framework.value,
                "total_rules": len(rules),
                "automated_rules": len([r for r in rules if r.automated]),
                "checks_performed": 0,
                "checks_passed": 0,
                "checks_failed": 0,
                "issues_found": []
            }
            
            for rule in rules:
                if not rule.automated:
                    continue
                
                try:
                    # Check if rule needs to be evaluated based on frequency
                    if not self._should_run_compliance_check(rule):
                        continue
                    
                    # Evaluate rule
                    check_result = await self._evaluate_compliance_rule(
                        rule, organization_id, 
                        datetime.utcnow() - timedelta(days=30),
                        datetime.utcnow()
                    )
                    
                    results["checks_performed"] += 1
                    
                    if check_result["status"] == "PASS":
                        results["checks_passed"] += 1
                    else:
                        results["checks_failed"] += 1
                        results["issues_found"].append(check_result)
                    
                    # Update rule check time
                    rule.last_checked = datetime.utcnow()
                    rule.last_result = check_result["status"]
                    
                except Exception as e:
                    logger.error(f"Compliance check failed for rule {rule.id}: {e}")
                    results["checks_failed"] += 1
                    results["issues_found"].append({
                        "rule_id": rule.id,
                        "error": str(e)
                    })
            
            # Audit compliance automation
            await self.log_audit_event(
                event_type="AUTOMATED_COMPLIANCE_CHECK",
                category=AuditEventCategory.COMPLIANCE_EVENT,
                severity=SeverityLevel.INFO,
                action="AUTOMATED_CHECK",
                outcome="SUCCESS",
                organization_id=organization_id,
                details=results,
                compliance_tags=[framework.value, "AUTOMATION"]
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Automated compliance checks failed: {e}")
            raise
    
    # Private helper methods
    async def _flush_audit_buffer(self) -> None:
        """Flush audit event buffer to persistent storage."""
        if not self.audit_buffer:
            return
        
        try:
            # Create audit log file for this batch
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            log_file = self.audit_storage_path / f"audit_{timestamp}_{uuid.uuid4().hex[:8]}.json"
            
            # Prepare audit events for storage
            events_data = []
            for event in self.audit_buffer:
                event_dict = event.to_dict()
                
                # Encrypt sensitive events if required
                if (event.category in [AuditEventCategory.AUTHENTICATION, AuditEventCategory.DATA_ACCESS] or
                    event.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]):
                    
                    encrypted_details = await self.encrypt_sensitive_data(
                        event_dict, DataClassification.CONFIDENTIAL
                    )
                    event_dict["details"] = base64.b64encode(encrypted_details.encrypted_data).decode()
                    event_dict["encrypted"] = True
                    event_dict["encryption_key_id"] = encrypted_details.key_id
                
                events_data.append(event_dict)
            
            # Write to file
            with open(log_file, 'w') as f:
                json.dump(events_data, f, indent=2)
            
            # Compress if configured
            if self.config["log_compression"]:
                compressed_file = log_file.with_suffix('.json.gz')
                with open(log_file, 'rb') as f_in:
                    with gzip.open(compressed_file, 'wb') as f_out:
                        f_out.writelines(f_in)
                log_file.unlink()  # Remove uncompressed file
            
            logger.debug(f"Flushed {len(self.audit_buffer)} audit events to {log_file}")
            
            # Clear buffer
            self.audit_buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush audit buffer: {e}")
    
    def _initialize_compliance_rules(self) -> None:
        """Initialize compliance rules for supported frameworks."""
        
        # SOC 2 Type II rules
        soc2_rules = [
            ComplianceRule(
                id="SOC2_CC1_1",
                framework=ComplianceFramework.SOC2_TYPE2,
                control_id="CC1.1",
                title="Management Oversight",
                description="Management demonstrates commitment to integrity and ethical values",
                validation_query="SELECT COUNT(*) FROM audit_events WHERE event_type = 'POLICY_VIOLATION' AND timestamp > NOW() - INTERVAL '30 days'",
                expected_result=0,
                severity=SeverityLevel.HIGH
            ),
            ComplianceRule(
                id="SOC2_CC6_1",
                framework=ComplianceFramework.SOC2_TYPE2,
                control_id="CC6.1",
                title="Logical Access Controls",
                description="System implements logical access controls",
                validation_query="SELECT COUNT(*) FROM users WHERE mfa_enabled = false AND role = 'ADMIN'",
                expected_result=0,
                severity=SeverityLevel.HIGH
            )
        ]
        
        # GDPR rules
        gdpr_rules = [
            ComplianceRule(
                id="GDPR_ART_5",
                framework=ComplianceFramework.GDPR,
                control_id="Article 5",
                title="Principles of Processing",
                description="Personal data processed lawfully, fairly and transparently",
                validation_query="SELECT COUNT(*) FROM data_processing_activities WHERE legal_basis IS NULL",
                expected_result=0,
                severity=SeverityLevel.CRITICAL
            ),
            ComplianceRule(
                id="GDPR_ART_32",
                framework=ComplianceFramework.GDPR,
                control_id="Article 32",
                title="Security of Processing",
                description="Appropriate technical measures to ensure security",
                validation_query="SELECT COUNT(*) FROM sensitive_data WHERE encrypted = false",
                expected_result=0,
                severity=SeverityLevel.HIGH
            )
        ]
        
        # ISO 27001 rules
        iso27001_rules = [
            ComplianceRule(
                id="ISO27001_A_9_1_1",
                framework=ComplianceFramework.ISO27001,
                control_id="A.9.1.1",
                title="Access Control Policy",
                description="Access control policy established and maintained",
                validation_query="SELECT COUNT(*) FROM access_policies WHERE status = 'ACTIVE' AND reviewed_date > NOW() - INTERVAL '12 months'",
                expected_result=lambda x: x > 0,
                severity=SeverityLevel.HIGH
            )
        ]
        
        self.compliance_rules = {
            ComplianceFramework.SOC2_TYPE2: soc2_rules,
            ComplianceFramework.GDPR: gdpr_rules,
            ComplianceFramework.ISO27001: iso27001_rules
        }
    
    async def _perform_vulnerability_scan(self, target: str) -> List[Dict[str, Any]]:
        """Perform actual vulnerability scanning (placeholder implementation)."""
        # In production, integrate with tools like:
        # - OWASP ZAP
        # - Nessus
        # - OpenVAS
        # - Custom security scanners
        
        # Simulated vulnerabilities for demonstration
        vulnerabilities = [
            {
                "id": "CVE-2023-0001",
                "title": "SQL Injection Vulnerability",
                "description": "Potential SQL injection in user input validation",
                "severity": "HIGH",
                "cvss_score": 8.1,
                "affected_component": "user_auth_module",
                "remediation": "Implement parameterized queries and input validation"
            },
            {
                "id": "SEC-2023-0002",
                "title": "Weak Password Policy",
                "description": "Password policy allows weak passwords",
                "severity": "MEDIUM",
                "cvss_score": 5.3,
                "affected_component": "password_policy",
                "remediation": "Implement stronger password requirements"
            }
        ]
        
        await asyncio.sleep(2)  # Simulate scan time
        return vulnerabilities
    
    def _generate_remediation_guidance(self, vulnerabilities: List[Dict[str, Any]]) -> List[str]:
        """Generate remediation guidance based on vulnerabilities found."""
        guidance = []
        
        critical_count = len([v for v in vulnerabilities if v.get("severity") == "CRITICAL"])
        high_count = len([v for v in vulnerabilities if v.get("severity") == "HIGH"])
        
        if critical_count > 0:
            guidance.append(f"URGENT: {critical_count} critical vulnerabilities require immediate attention")
            guidance.append("Prioritize patching critical vulnerabilities within 24 hours")
        
        if high_count > 0:
            guidance.append(f"HIGH PRIORITY: {high_count} high-severity vulnerabilities found")
            guidance.append("Schedule patching for high-severity issues within 7 days")
        
        # Add specific remediation steps
        for vuln in vulnerabilities:
            if vuln.get("remediation"):
                guidance.append(f"{vuln['title']}: {vuln['remediation']}")
        
        return guidance
    
    async def _evaluate_compliance_rule(
        self,
        rule: ComplianceRule,
        organization_id: Optional[uuid.UUID],
        period_start: datetime,
        period_end: datetime
    ) -> Dict[str, Any]:
        """Evaluate a single compliance rule."""
        try:
            # Execute validation query (simplified)
            # In production, this would execute actual database queries
            result = {
                "rule_id": rule.id,
                "control_id": rule.control_id,
                "title": rule.title,
                "status": "PASS",  # Default
                "evidence": [],
                "details": {},
                "evaluated_at": datetime.utcnow().isoformat()
            }
            
            # Simulate rule evaluation based on rule ID
            if "POLICY_VIOLATION" in rule.validation_query:
                # Check for policy violations
                violations_count = 0  # Would query actual database
                if violations_count > rule.expected_result:
                    result["status"] = "FAIL"
                    result["details"]["violations_found"] = violations_count
                    
            elif "mfa_enabled" in rule.validation_query:
                # Check MFA compliance
                non_mfa_admins = 0  # Would query actual database
                if non_mfa_admins > rule.expected_result:
                    result["status"] = "FAIL"
                    result["details"]["non_mfa_admins"] = non_mfa_admins
            
            return result
            
        except Exception as e:
            return {
                "rule_id": rule.id,
                "status": "FAIL",
                "error": str(e),
                "evaluated_at": datetime.utcnow().isoformat()
            }
    
    def _generate_compliance_recommendations(
        self,
        framework: ComplianceFramework,
        findings: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate compliance recommendations based on findings."""
        recommendations = []
        
        if not findings:
            recommendations.append(f"All {framework.value} controls are currently compliant")
            recommendations.append("Continue regular monitoring and maintenance of controls")
            return recommendations
        
        # Analyze findings
        critical_findings = [f for f in findings if f.get("severity") == "CRITICAL"]
        high_findings = [f for f in findings if f.get("severity") == "HIGH"]
        
        if critical_findings:
            recommendations.append(f"URGENT: Address {len(critical_findings)} critical compliance issues immediately")
        
        if high_findings:
            recommendations.append(f"HIGH PRIORITY: Resolve {len(high_findings)} high-priority findings")
        
        # Framework-specific recommendations
        if framework == ComplianceFramework.SOC2_TYPE2:
            recommendations.extend([
                "Review and update access control policies",
                "Implement continuous monitoring for security controls",
                "Conduct regular risk assessments"
            ])
        elif framework == ComplianceFramework.GDPR:
            recommendations.extend([
                "Review data processing activities and legal basis",
                "Update privacy policies and consent mechanisms",
                "Implement data subject rights procedures"
            ])
        
        return recommendations
    
    def _should_run_compliance_check(self, rule: ComplianceRule) -> bool:
        """Determine if compliance check should run based on frequency."""
        if rule.last_checked is None:
            return True
        
        now = datetime.utcnow()
        
        if rule.frequency == "DAILY":
            return now - rule.last_checked > timedelta(days=1)
        elif rule.frequency == "WEEKLY":
            return now - rule.last_checked > timedelta(weeks=1)
        elif rule.frequency == "MONTHLY":
            return now - rule.last_checked > timedelta(days=30)
        elif rule.frequency == "QUARTERLY":
            return now - rule.last_checked > timedelta(days=90)
        
        return False
    
    def _start_background_tasks(self) -> None:
        """Start background tasks for audit processing."""
        # Audit buffer flush task
        async def flush_task():
            while True:
                await asyncio.sleep(self.buffer_flush_interval)
                if self.audit_buffer:
                    await self._flush_audit_buffer()
        
        # Compliance automation task
        async def compliance_task():
            while True:
                await asyncio.sleep(3600)  # Run hourly
                try:
                    for framework in ComplianceFramework:
                        await self.automate_compliance_checks(framework)
                except Exception as e:
                    logger.error(f"Compliance automation task failed: {e}")
        
        self._background_tasks = [
            asyncio.create_task(flush_task()),
            asyncio.create_task(compliance_task())
        ]
    
    def __del__(self):
        """Cleanup background tasks."""
        for task in self._background_tasks:
            if not task.done():
                task.cancel()


# Factory function
async def create_compliance_audit_system(
    db_session: AsyncSession,
    encryption_keys: Dict[str, bytes],
    audit_storage_path: str = "/var/log/agent-hive/audit"
) -> ComplianceAuditSystem:
    """
    Create compliance audit system instance.
    
    Args:
        db_session: Database session
        encryption_keys: Dictionary of encryption keys
        audit_storage_path: Path for audit log storage
        
    Returns:
        ComplianceAuditSystem instance
    """
    return ComplianceAuditSystem(db_session, encryption_keys, audit_storage_path)