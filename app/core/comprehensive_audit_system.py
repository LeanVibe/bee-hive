"""
Comprehensive Audit Logging System for Enterprise Compliance.

Provides enterprise-grade audit logging with support for SOC 2, ISO 27001,
and NIST compliance requirements. Features real-time monitoring, integrity
verification, and automated compliance reporting.

Features:
- Real-time audit event capture
- Compliance framework support (SOC 2, ISO 27001, NIST)
- Integrity verification and tamper detection
- Automated compliance reporting
- Event correlation and analysis
- Performance-optimized logging
- Data retention and archival
"""

import asyncio
import hashlib
import hmac
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc, text
from sqlalchemy.orm import selectinload

from .redis import RedisClient
from .database import get_session
from ..models.security import SecurityAuditLog, SecurityEvent, AgentIdentity, AgentRoleAssignment
from ..schemas.security import SecurityEventSeverityEnum

logger = structlog.get_logger()


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    NIST = "nist"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"


class AuditEventCategory(Enum):
    """Audit event categories for classification."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    CONFIGURATION_CHANGE = "configuration_change"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SYSTEM_ADMINISTRATION = "system_administration"
    USER_MANAGEMENT = "user_management"
    DATA_MODIFICATION = "data_modification"
    SECURITY_INCIDENT = "security_incident"
    COMPLIANCE_VIOLATION = "compliance_violation"


class RiskLevel(Enum):
    """Risk levels for audit events."""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditContext:
    """Comprehensive audit context for event logging."""
    # Core identifiers
    agent_id: Optional[uuid.UUID] = None
    user_id: Optional[str] = None
    session_id: Optional[uuid.UUID] = None
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    # Action details
    action: str = ""
    resource: Optional[str] = None
    resource_id: Optional[str] = None
    resource_type: Optional[str] = None
    
    # Request context
    method: Optional[str] = None
    endpoint: Optional[str] = None
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    geo_location: Optional[str] = None
    
    # Security context
    authentication_method: Optional[str] = None
    authorization_result: Optional[str] = None
    permissions_checked: List[str] = field(default_factory=list)
    roles_used: List[str] = field(default_factory=list)
    
    # Data context
    request_data: Optional[Dict[str, Any]] = None
    response_data: Optional[Dict[str, Any]] = None
    sensitive_fields: List[str] = field(default_factory=list)
    data_classification: Optional[str] = None
    
    # Outcome
    success: bool = True
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    http_status_code: Optional[int] = None
    
    # Performance
    start_time: Optional[datetime] = None
    duration_ms: Optional[int] = None
    
    # Compliance
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    retention_period_days: int = 2555  # 7 years default
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceRule:
    """Compliance rule definition."""
    framework: ComplianceFramework
    rule_id: str
    description: str
    event_categories: List[AuditEventCategory]
    required_fields: List[str]
    retention_days: int
    real_time_monitoring: bool = False
    alert_on_violation: bool = False
    
    def matches_event(self, category: AuditEventCategory, fields: Dict[str, Any]) -> bool:
        """Check if event matches this compliance rule."""
        if category not in self.event_categories:
            return False
        
        # Check required fields
        for field in self.required_fields:
            if field not in fields or fields[field] is None:
                return False
        
        return True


class ComprehensiveAuditSystem:
    """
    Comprehensive Audit Logging System.
    
    Provides enterprise-grade audit logging with compliance framework support,
    integrity verification, and automated monitoring.
    """
    
    # Predefined compliance rules
    COMPLIANCE_RULES = {
        ComplianceFramework.SOC2: [
            ComplianceRule(
                framework=ComplianceFramework.SOC2,
                rule_id="CC6.1",
                description="Logical and physical access controls",
                event_categories=[AuditEventCategory.AUTHENTICATION, AuditEventCategory.AUTHORIZATION],
                required_fields=["user_id", "timestamp", "success", "client_ip"],
                retention_days=365,
                real_time_monitoring=True,
                alert_on_violation=True
            ),
            ComplianceRule(
                framework=ComplianceFramework.SOC2,
                rule_id="CC6.2",
                description="Access authorization and modification",
                event_categories=[AuditEventCategory.PRIVILEGE_ESCALATION, AuditEventCategory.USER_MANAGEMENT],
                required_fields=["user_id", "action", "resource", "timestamp", "authorized_by"],
                retention_days=2555,  # 7 years
                real_time_monitoring=True,
                alert_on_violation=True
            ),
            ComplianceRule(
                framework=ComplianceFramework.SOC2,
                rule_id="CC6.3",
                description="Network security monitoring",
                event_categories=[AuditEventCategory.SECURITY_INCIDENT, AuditEventCategory.DATA_ACCESS],
                required_fields=["client_ip", "action", "timestamp", "success"],
                retention_days=365,
                real_time_monitoring=True
            )
        ],
        ComplianceFramework.ISO27001: [
            ComplianceRule(
                framework=ComplianceFramework.ISO27001,
                rule_id="A.12.4.1",
                description="Event logging",
                event_categories=list(AuditEventCategory),
                required_fields=["timestamp", "user_id", "action", "outcome"],
                retention_days=2555,
                real_time_monitoring=False
            ),
            ComplianceRule(
                framework=ComplianceFramework.ISO27001,
                rule_id="A.9.2.6",
                description="Access rights review",
                event_categories=[AuditEventCategory.AUTHORIZATION, AuditEventCategory.PRIVILEGE_ESCALATION],
                required_fields=["user_id", "permissions_checked", "authorization_result", "timestamp"],
                retention_days=1095,  # 3 years
                real_time_monitoring=True
            )
        ],
        ComplianceFramework.NIST: [
            ComplianceRule(
                framework=ComplianceFramework.NIST,
                rule_id="AU-2",
                description="Audit Events",
                event_categories=list(AuditEventCategory),
                required_fields=["timestamp", "event_type", "user_id", "outcome"],
                retention_days=2555,
                real_time_monitoring=False
            ),
            ComplianceRule(
                framework=ComplianceFramework.NIST,
                rule_id="AU-3",
                description="Content of Audit Records",
                event_categories=list(AuditEventCategory),
                required_fields=["timestamp", "event_type", "user_id", "outcome", "source", "object"],
                retention_days=2555,
                real_time_monitoring=False
            )
        ]
    }
    
    def __init__(
        self,
        db_session: AsyncSession,
        redis_client: RedisClient,
        integrity_key: Optional[str] = None,
        enabled_frameworks: Optional[List[ComplianceFramework]] = None
    ):
        """
        Initialize Comprehensive Audit System.
        
        Args:
            db_session: Database session
            redis_client: Redis client for caching and streaming
            integrity_key: HMAC key for log integrity verification
            enabled_frameworks: List of enabled compliance frameworks
        """
        self.db = db_session
        self.redis = redis_client
        self.integrity_key = integrity_key or "default-audit-integrity-key"
        self.enabled_frameworks = enabled_frameworks or [ComplianceFramework.SOC2, ComplianceFramework.ISO27001]
        
        # Performance metrics
        self.metrics = {
            "events_logged": 0,
            "events_by_category": {},
            "compliance_violations": 0,
            "integrity_failures": 0,
            "avg_logging_time_ms": 0.0,
            "real_time_alerts_sent": 0,
            "retention_policy_applied": 0
        }
        
        # Cache keys
        self._event_stream_key = "audit:events:stream"
        self._compliance_cache_prefix = "audit:compliance:"
        self._integrity_cache_prefix = "audit:integrity:"
        self._metrics_key = "audit:metrics"
        
        # Real-time monitoring
        self._real_time_monitors: List[Callable] = []
        
        # Compliance rules
        self.active_rules = []
        for framework in self.enabled_frameworks:
            self.active_rules.extend(self.COMPLIANCE_RULES.get(framework, []))
    
    async def log_audit_event(
        self,
        context: AuditContext,
        category: AuditEventCategory,
        compliance_frameworks: Optional[List[ComplianceFramework]] = None
    ) -> str:
        """
        Log comprehensive audit event.
        
        Args:
            context: Audit context
            category: Event category
            compliance_frameworks: Specific compliance frameworks
            
        Returns:
            Audit event ID
        """
        start_time = time.time()
        
        try:
            # Generate event ID
            event_id = str(uuid.uuid4())
            
            # Set default compliance frameworks
            if compliance_frameworks:
                context.compliance_frameworks = compliance_frameworks
            elif not context.compliance_frameworks:
                context.compliance_frameworks = self.enabled_frameworks
            
            # Calculate risk score
            risk_score = await self._calculate_risk_score(context, category)
            
            # Generate integrity signature
            log_signature = self._generate_integrity_signature(context, event_id)
            
            # Create audit log entry
            audit_log = SecurityAuditLog(
                agent_id=context.agent_id,
                human_controller=context.user_id or "system",
                session_id=context.session_id,
                request_id=context.request_id,
                action=context.action,
                resource=context.resource,
                resource_id=context.resource_id,
                method=context.method,
                endpoint=context.endpoint,
                request_data=self._sanitize_sensitive_data(context.request_data, context.sensitive_fields),
                response_data=self._sanitize_sensitive_data(context.response_data, context.sensitive_fields),
                ip_address=context.client_ip,
                user_agent=context.user_agent,
                geo_location=context.geo_location,
                success=context.success,
                http_status_code=context.http_status_code,
                error_message=context.error_message,
                error_code=context.error_code,
                duration_ms=context.duration_ms,
                permission_checked=",".join(context.permissions_checked) if context.permissions_checked else None,
                authorization_result=context.authorization_result,
                risk_score=risk_score,
                security_labels=[category.value] + [f.value for f in context.compliance_frameworks],
                correlation_id=context.correlation_id,
                log_signature=log_signature,
                agent_metadata={
                    "event_id": event_id,
                    "category": category.value,
                    "compliance_frameworks": [f.value for f in context.compliance_frameworks],
                    "retention_period_days": context.retention_period_days,
                    "authentication_method": context.authentication_method,
                    "roles_used": context.roles_used,
                    "data_classification": context.data_classification,
                    **context.metadata
                }
            )
            
            # Store in database
            self.db.add(audit_log)
            await self.db.commit()
            await self.db.refresh(audit_log)
            
            # Stream to Redis for real-time processing
            await self._stream_audit_event(audit_log, category)
            
            # Check compliance rules
            compliance_violations = await self._check_compliance_rules(context, category, audit_log)
            
            # Real-time monitoring
            if any(rule.real_time_monitoring for rule in self.active_rules if rule.matches_event(category, context.__dict__)):
                await self._trigger_real_time_monitoring(audit_log, category, compliance_violations)
            
            # Update metrics
            self._update_metrics(category, time.time() - start_time, compliance_violations)
            
            logger.info("Audit event logged", 
                       event_id=event_id, 
                       category=category.value, 
                       success=context.success,
                       risk_score=risk_score)
            
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}", 
                        category=category.value,
                        action=context.action)
            raise
    
    async def log_security_event(
        self,
        event_type: str,
        severity: SecurityEventSeverityEnum,
        description: str,
        agent_id: Optional[uuid.UUID] = None,
        details: Optional[Dict[str, Any]] = None,
        risk_score: Optional[float] = None,
        source_ip: Optional[str] = None,
        auto_detected: bool = True
    ) -> str:
        """
        Log security event for monitoring and incident response.
        
        Args:
            event_type: Type of security event
            severity: Event severity level
            description: Event description
            agent_id: Associated agent ID
            details: Additional event details
            risk_score: Risk score (0.0-1.0)
            source_ip: Source IP address
            auto_detected: Whether event was auto-detected
            
        Returns:
            Security event ID
        """
        try:
            security_event = SecurityEvent(
                event_type=event_type,
                severity=severity.value,
                agent_id=agent_id,
                source_ip=source_ip,
                description=description,
                details=details or {},
                risk_score=risk_score,
                auto_detected=auto_detected
            )
            
            self.db.add(security_event)
            await self.db.commit()
            await self.db.refresh(security_event)
            
            # Stream for real-time alerting
            await self._stream_security_event(security_event)
            
            # Trigger alerts for high-severity events
            if severity in [SecurityEventSeverityEnum.HIGH, SecurityEventSeverityEnum.CRITICAL]:
                await self._trigger_security_alert(security_event)
            
            return str(security_event.id)
            
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")
            raise
    
    async def verify_log_integrity(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        sample_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Verify audit log integrity using signatures.
        
        Args:
            start_date: Start date for verification
            end_date: End date for verification
            sample_size: Number of logs to sample
            
        Returns:
            Integrity verification results
        """
        try:
            # Build query
            query = select(SecurityAuditLog).where(SecurityAuditLog.log_signature.is_not(None))
            
            if start_date:
                query = query.where(SecurityAuditLog.timestamp >= start_date)
            if end_date:
                query = query.where(SecurityAuditLog.timestamp <= end_date)
            
            query = query.order_by(func.random()).limit(sample_size)
            
            result = await self.db.execute(query)
            logs = result.scalars().all()
            
            # Verify signatures
            verification_results = {
                "total_checked": len(logs),
                "valid_signatures": 0,
                "invalid_signatures": 0,
                "missing_signatures": 0,
                "integrity_failures": []
            }
            
            for log in logs:
                if not log.log_signature:
                    verification_results["missing_signatures"] += 1
                    continue
                
                # Reconstruct context for verification
                context = AuditContext(
                    agent_id=log.agent_id,
                    user_id=log.human_controller,
                    session_id=log.session_id,
                    request_id=log.request_id,
                    action=log.action,
                    resource=log.resource,
                    resource_id=log.resource_id,
                    method=log.method,
                    endpoint=log.endpoint,
                    client_ip=log.ip_address,
                    user_agent=log.user_agent,
                    success=log.success,
                    error_message=log.error_message,
                    error_code=log.error_code,
                    correlation_id=log.correlation_id,
                    metadata=log.agent_metadata or {}
                )
                
                # Verify signature
                expected_signature = self._generate_integrity_signature(
                    context, log.agent_metadata.get("event_id", str(log.id))
                )
                
                if hmac.compare_digest(log.log_signature, expected_signature):
                    verification_results["valid_signatures"] += 1
                else:
                    verification_results["invalid_signatures"] += 1
                    verification_results["integrity_failures"].append({
                        "log_id": str(log.id),
                        "timestamp": log.timestamp.isoformat(),
                        "action": log.action,
                        "expected_signature": expected_signature,
                        "actual_signature": log.log_signature
                    })
            
            # Calculate integrity percentage
            total_with_signatures = verification_results["valid_signatures"] + verification_results["invalid_signatures"]
            if total_with_signatures > 0:
                verification_results["integrity_percentage"] = (
                    verification_results["valid_signatures"] / total_with_signatures * 100
                )
            else:
                verification_results["integrity_percentage"] = 0.0
            
            # Update metrics
            self.metrics["integrity_failures"] += verification_results["invalid_signatures"]
            
            return verification_results
            
        except Exception as e:
            logger.error(f"Log integrity verification failed: {e}")
            raise
    
    async def generate_compliance_report(
        self,
        framework: ComplianceFramework,
        start_date: datetime,
        end_date: datetime,
        include_violations: bool = True
    ) -> Dict[str, Any]:
        """
        Generate compliance report for specified framework.
        
        Args:
            framework: Compliance framework
            start_date: Report start date
            end_date: Report end date
            include_violations: Include compliance violations
            
        Returns:
            Compliance report
        """
        try:
            framework_rules = [rule for rule in self.active_rules if rule.framework == framework]
            
            report = {
                "framework": framework.value,
                "report_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "rules_evaluated": len(framework_rules),
                "rule_compliance": {},
                "overall_compliance_score": 0.0,
                "violations": [],
                "recommendations": []
            }
            
            total_compliance_score = 0.0
            
            for rule in framework_rules:
                rule_compliance = await self._evaluate_rule_compliance(rule, start_date, end_date)
                report["rule_compliance"][rule.rule_id] = rule_compliance
                total_compliance_score += rule_compliance["compliance_score"]
            
            # Calculate overall compliance score
            if framework_rules:
                report["overall_compliance_score"] = total_compliance_score / len(framework_rules)
            
            # Add violations if requested
            if include_violations:
                violations = await self._get_compliance_violations(framework, start_date, end_date)
                report["violations"] = violations
            
            # Generate recommendations
            report["recommendations"] = await self._generate_compliance_recommendations(framework, report)
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate compliance report: {e}")
            raise
    
    async def get_audit_analytics(
        self,
        start_date: datetime,
        end_date: datetime,
        group_by: str = "day"
    ) -> Dict[str, Any]:
        """
        Get audit analytics and insights.
        
        Args:
            start_date: Start date for analytics
            end_date: End date for analytics
            group_by: Grouping interval (hour, day, week)
            
        Returns:
            Audit analytics data
        """
        try:
            # Base query
            base_query = select(SecurityAuditLog).where(
                and_(
                    SecurityAuditLog.timestamp >= start_date,
                    SecurityAuditLog.timestamp <= end_date
                )
            )
            
            # Get total events
            total_result = await self.db.execute(select(func.count()).select_from(base_query.subquery()))
            total_events = total_result.scalar()
            
            # Get success/failure breakdown
            success_result = await self.db.execute(
                select(SecurityAuditLog.success, func.count()).
                select_from(base_query.subquery()).
                group_by(SecurityAuditLog.success)
            )
            success_breakdown = {str(row[0]): row[1] for row in success_result}
            
            # Get top actions
            action_result = await self.db.execute(
                select(SecurityAuditLog.action, func.count()).
                select_from(base_query.subquery()).
                group_by(SecurityAuditLog.action).
                order_by(desc(func.count())).
                limit(10)
            )
            top_actions = [{"action": row[0], "count": row[1]} for row in action_result]
            
            # Get top users
            user_result = await self.db.execute(
                select(SecurityAuditLog.human_controller, func.count()).
                select_from(base_query.subquery()).
                group_by(SecurityAuditLog.human_controller).
                order_by(desc(func.count())).
                limit(10)
            )
            top_users = [{"user": row[0], "count": row[1]} for row in user_result]
            
            # Get risk score distribution
            risk_result = await self.db.execute(
                select(
                    func.case(
                        (SecurityAuditLog.risk_score <= 0.2, "Low"),
                        (SecurityAuditLog.risk_score <= 0.5, "Medium"),
                        (SecurityAuditLog.risk_score <= 0.8, "High"),
                        else_="Critical"
                    ).label("risk_level"),
                    func.count()
                ).
                select_from(base_query.subquery()).
                where(SecurityAuditLog.risk_score.is_not(None)).
                group_by("risk_level")
            )
            risk_distribution = {row[0]: row[1] for row in risk_result}
            
            # Get error patterns
            error_result = await self.db.execute(
                select(SecurityAuditLog.error_code, func.count()).
                select_from(base_query.subquery()).
                where(SecurityAuditLog.success == False).
                group_by(SecurityAuditLog.error_code).
                order_by(desc(func.count())).
                limit(10)
            )
            error_patterns = [{"error_code": row[0], "count": row[1]} for row in error_result if row[0]]
            
            return {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "summary": {
                    "total_events": total_events,
                    "success_rate": success_breakdown.get("True", 0) / max(total_events, 1) * 100,
                    "failure_rate": success_breakdown.get("False", 0) / max(total_events, 1) * 100
                },
                "top_actions": top_actions,
                "top_users": top_users,
                "risk_distribution": risk_distribution,
                "error_patterns": error_patterns,
                "compliance_frameworks": [f.value for f in self.enabled_frameworks]
            }
            
        except Exception as e:
            logger.error(f"Failed to get audit analytics: {e}")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get audit system metrics."""
        return {
            "audit_system_metrics": self.metrics.copy(),
            "enabled_frameworks": [f.value for f in self.enabled_frameworks],
            "active_rules": len(self.active_rules),
            "real_time_monitors": len(self._real_time_monitors)
        }
    
    # Private helper methods
    
    async def _calculate_risk_score(self, context: AuditContext, category: AuditEventCategory) -> float:
        """Calculate risk score for audit event."""
        risk_factors = {
            "failed_action": 0.3 if not context.success else 0.0,
            "privileged_action": 0.2 if category in [
                AuditEventCategory.PRIVILEGE_ESCALATION,
                AuditEventCategory.SYSTEM_ADMINISTRATION,
                AuditEventCategory.USER_MANAGEMENT
            ] else 0.0,
            "sensitive_data": 0.2 if context.data_classification in ["confidential", "restricted"] else 0.0,
            "off_hours": 0.1 if self._is_off_hours() else 0.0,
            "high_risk_error": 0.2 if context.error_code in ["AUTH_FAILURE", "PERMISSION_DENIED"] else 0.0,
            "bulk_operation": 0.1 if "bulk" in context.action.lower() else 0.0
        }
        
        return min(1.0, sum(risk_factors.values()))
    
    def _generate_integrity_signature(self, context: AuditContext, event_id: str) -> str:
        """Generate HMAC signature for log integrity."""
        # Create signature payload
        payload_data = {
            "event_id": event_id,
            "user_id": context.user_id,
            "action": context.action,
            "resource": context.resource,
            "timestamp": context.start_time.isoformat() if context.start_time else datetime.utcnow().isoformat(),
            "success": context.success,
            "client_ip": context.client_ip
        }
        
        payload_json = json.dumps(payload_data, sort_keys=True)
        signature = hmac.new(
            self.integrity_key.encode(),
            payload_json.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _sanitize_sensitive_data(
        self,
        data: Optional[Dict[str, Any]],
        sensitive_fields: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Sanitize sensitive data for logging."""
        if not data:
            return data
        
        sanitized = data.copy()
        
        for field in sensitive_fields:
            if field in sanitized:
                if isinstance(sanitized[field], str):
                    # Keep first and last 2 characters, mask the rest
                    value = sanitized[field]
                    if len(value) > 4:
                        sanitized[field] = value[:2] + "*" * (len(value) - 4) + value[-2:]
                    else:
                        sanitized[field] = "*" * len(value)
                else:
                    sanitized[field] = "[REDACTED]"
        
        return sanitized
    
    async def _stream_audit_event(self, audit_log: SecurityAuditLog, category: AuditEventCategory) -> None:
        """Stream audit event to Redis for real-time processing."""
        try:
            event_data = {
                "id": str(audit_log.id),
                "category": category.value,
                "action": audit_log.action,
                "user_id": audit_log.human_controller,
                "success": audit_log.success,
                "risk_score": audit_log.risk_score,
                "timestamp": audit_log.timestamp.isoformat()
            }
            
            await self.redis.xadd(self._event_stream_key, event_data)
            
        except Exception as e:
            logger.error(f"Failed to stream audit event: {e}")
    
    async def _stream_security_event(self, security_event: SecurityEvent) -> None:
        """Stream security event for real-time alerting."""
        try:
            event_data = {
                "id": str(security_event.id),
                "event_type": security_event.event_type,
                "severity": security_event.severity,
                "description": security_event.description,
                "risk_score": security_event.risk_score,
                "timestamp": security_event.timestamp.isoformat()
            }
            
            await self.redis.xadd("security:events:stream", event_data)
            
        except Exception as e:
            logger.error(f"Failed to stream security event: {e}")
    
    async def _check_compliance_rules(
        self,
        context: AuditContext,
        category: AuditEventCategory,
        audit_log: SecurityAuditLog
    ) -> List[str]:
        """Check compliance rules and return violations."""
        violations = []
        
        for rule in self.active_rules:
            if rule.matches_event(category, context.__dict__):
                # Check if all required fields are present
                missing_fields = []
                for field in rule.required_fields:
                    if not hasattr(context, field) or getattr(context, field) is None:
                        missing_fields.append(field)
                
                if missing_fields:
                    violation = f"Rule {rule.rule_id}: Missing required fields: {', '.join(missing_fields)}"
                    violations.append(violation)
                    
                    # Log compliance violation
                    await self.log_security_event(
                        event_type="compliance_violation",
                        severity=SecurityEventSeverityEnum.MEDIUM,
                        description=violation,
                        details={
                            "rule_id": rule.rule_id,
                            "framework": rule.framework.value,
                            "missing_fields": missing_fields,
                            "audit_log_id": str(audit_log.id)
                        }
                    )
        
        return violations
    
    async def _trigger_real_time_monitoring(
        self,
        audit_log: SecurityAuditLog,
        category: AuditEventCategory,
        violations: List[str]
    ) -> None:
        """Trigger real-time monitoring alerts."""
        try:
            for monitor in self._real_time_monitors:
                await monitor(audit_log, category, violations)
            
            self.metrics["real_time_alerts_sent"] += 1
            
        except Exception as e:
            logger.error(f"Real-time monitoring failed: {e}")
    
    async def _trigger_security_alert(self, security_event: SecurityEvent) -> None:
        """Trigger security alert for high-severity events."""
        try:
            alert_data = {
                "event_id": str(security_event.id),
                "event_type": security_event.event_type,
                "severity": security_event.severity,
                "description": security_event.description,
                "risk_score": security_event.risk_score,
                "timestamp": security_event.timestamp.isoformat(),
                "requires_immediate_attention": security_event.severity == SecurityEventSeverityEnum.CRITICAL.value
            }
            
            # Store alert for external processing
            await self.redis.lpush("security:alerts", json.dumps(alert_data))
            
        except Exception as e:
            logger.error(f"Failed to trigger security alert: {e}")
    
    async def _evaluate_rule_compliance(
        self,
        rule: ComplianceRule,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Evaluate compliance for a specific rule."""
        # This would implement rule-specific compliance evaluation
        # For now, return a basic structure
        return {
            "rule_id": rule.rule_id,
            "description": rule.description,
            "compliance_score": 0.95,  # Placeholder
            "events_evaluated": 0,
            "violations_found": 0,
            "recommendations": []
        }
    
    async def _get_compliance_violations(
        self,
        framework: ComplianceFramework,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get compliance violations for framework."""
        # Query security events for compliance violations
        query = select(SecurityEvent).where(
            and_(
                SecurityEvent.event_type == "compliance_violation",
                SecurityEvent.timestamp >= start_date,
                SecurityEvent.timestamp <= end_date
            )
        )
        
        result = await self.db.execute(query)
        violations = result.scalars().all()
        
        # Filter by framework
        framework_violations = []
        for violation in violations:
            details = violation.details or {}
            if details.get("framework") == framework.value:
                framework_violations.append(violation.to_dict())
        
        return framework_violations
    
    async def _generate_compliance_recommendations(
        self,
        framework: ComplianceFramework,
        report: Dict[str, Any]
    ) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []
        
        compliance_score = report["overall_compliance_score"]
        
        if compliance_score < 0.9:
            recommendations.append("Improve audit log completeness to meet compliance requirements")
        
        if report["violations"]:
            recommendations.append("Address identified compliance violations promptly")
        
        if framework == ComplianceFramework.SOC2:
            recommendations.extend([
                "Implement real-time monitoring for privileged access",
                "Ensure all access changes are properly authorized and documented"
            ])
        
        return recommendations
    
    def _is_off_hours(self) -> bool:
        """Check if current time is outside business hours."""
        current_hour = datetime.utcnow().hour
        return current_hour < 9 or current_hour > 17
    
    def _update_metrics(
        self,
        category: AuditEventCategory,
        processing_time: float,
        violations: List[str]
    ) -> None:
        """Update audit system metrics."""
        self.metrics["events_logged"] += 1
        
        category_key = category.value
        self.metrics["events_by_category"][category_key] = (
            self.metrics["events_by_category"].get(category_key, 0) + 1
        )
        
        self.metrics["compliance_violations"] += len(violations)
        
        # Update average processing time
        current_avg = self.metrics["avg_logging_time_ms"]
        total_events = self.metrics["events_logged"]
        processing_time_ms = processing_time * 1000
        
        self.metrics["avg_logging_time_ms"] = (
            (current_avg * (total_events - 1) + processing_time_ms) / total_events
        )
    
    def add_real_time_monitor(self, monitor: Callable) -> None:
        """Add real-time monitoring callback."""
        self._real_time_monitors.append(monitor)
    
    async def cleanup(self) -> None:
        """Cleanup resources and store final metrics."""
        try:
            await self.redis.set_with_expiry(
                self._metrics_key,
                json.dumps(self.get_metrics()),
                ttl=86400
            )
        except Exception as e:
            logger.error(f"Failed to store final audit metrics: {e}")


# Factory function
async def create_comprehensive_audit_system(
    db_session: AsyncSession,
    redis_client: RedisClient,
    integrity_key: Optional[str] = None,
    enabled_frameworks: Optional[List[ComplianceFramework]] = None
) -> ComprehensiveAuditSystem:
    """
    Create Comprehensive Audit System instance.
    
    Args:
        db_session: Database session
        redis_client: Redis client
        integrity_key: HMAC key for integrity verification
        enabled_frameworks: List of enabled compliance frameworks
        
    Returns:
        ComprehensiveAuditSystem instance
    """
    return ComprehensiveAuditSystem(
        db_session,
        redis_client,
        integrity_key,
        enabled_frameworks
    )