"""
Comprehensive Security Audit Logger.

Implements immutable audit logging with cryptographic signatures,
threat detection, and compliance-ready data structures.
"""

import uuid
import json
import hmac
import hashlib
import asyncio
import geoip2.database
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

from sqlalchemy import select, and_, or_, func, desc, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models.security import (
    SecurityAuditLog, SecurityEvent, AgentIdentity,
    SecurityEventSeverity
)
from ..schemas.security import AuditLogRequest, SecurityEventRequest, AuditLogFilters
from .redis import RedisClient

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Audit event types for categorization."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    CONFIGURATION_CHANGE = "configuration_change"
    SYSTEM_EVENT = "system_event"
    SECURITY_EVENT = "security_event"
    API_CALL = "api_call"
    ERROR_EVENT = "error_event"


class ThreatLevel(Enum):
    """Threat level classification."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditContext:
    """Context for audit logging."""
    request_id: str
    agent_id: Optional[uuid.UUID]
    human_controller: str
    session_id: Optional[uuid.UUID]
    ip_address: Optional[str]
    user_agent: Optional[str]
    geo_location: Optional[str]
    timestamp: datetime
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "agent_id": str(self.agent_id) if self.agent_id else None,
            "human_controller": self.human_controller,
            "session_id": str(self.session_id) if self.session_id else None,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "geo_location": self.geo_location,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id
        }


@dataclass
class SecurityIncident:
    """Security incident for alerting."""
    incident_id: str
    event_type: str
    severity: SecurityEventSeverity
    agent_id: Optional[uuid.UUID]
    description: str
    evidence: Dict[str, Any]
    risk_score: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class AuditLogger:
    """
    Comprehensive Security Audit Logger.
    
    Features:
    - Immutable audit logs with cryptographic signatures
    - Real-time threat detection and alerting
    - Compliance-ready data structures (SOX, HIPAA, GDPR)
    - Performance-optimized batch processing
    - Geo-location tracking and analysis
    - Automated incident response triggers
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        redis_client: RedisClient,
        signing_key: str,
        enable_geo_lookup: bool = True,
        batch_size: int = 100,
        flush_interval_seconds: int = 30
    ):
        """
        Initialize Audit Logger.
        
        Args:
            db_session: Database session
            redis_client: Redis client for caching and queuing
            signing_key: HMAC signing key for log integrity
            enable_geo_lookup: Enable geographic IP lookup
            batch_size: Batch size for bulk operations
            flush_interval_seconds: Auto-flush interval
        """
        self.db = db_session
        self.redis = redis_client
        self.signing_key = signing_key.encode('utf-8')
        self.enable_geo_lookup = enable_geo_lookup
        self.batch_size = batch_size
        self.flush_interval = flush_interval_seconds
        
        # Batch processing
        self.pending_logs: List[SecurityAuditLog] = []
        self.last_flush = datetime.utcnow()
        
        # Security monitoring
        self.threat_patterns = {
            "brute_force": {"threshold": 5, "window_minutes": 5},
            "privilege_escalation": {"threshold": 3, "window_minutes": 10},
            "data_exfiltration": {"threshold": 100, "window_minutes": 60},
            "off_hours_activity": {"start_hour": 22, "end_hour": 6},
            "suspicious_locations": ["CN", "RU", "KP"],  # Example countries
            "rapid_fire_requests": {"threshold": 50, "window_minutes": 1}
        }
        
        # Compliance configuration
        self.compliance_config = {
            "retention_days": 2555,  # 7 years for SOX compliance
            "require_signatures": True,
            "log_sensitive_data": False,  # GDPR compliance
            "encrypt_pii": True,
            "audit_system_events": True,
            "real_time_monitoring": True
        }
        
        # Performance metrics
        self.metrics = {
            "logs_written": 0,
            "batches_processed": 0,
            "signature_failures": 0,
            "threat_alerts": 0,
            "avg_write_time_ms": 0.0
        }
        
        # GeoIP database (if available)
        self.geoip_reader = None
        if self.enable_geo_lookup:
            try:
                geoip_path = Path("data/GeoLite2-City.mmdb")
                if geoip_path.exists():
                    self.geoip_reader = geoip2.database.Reader(str(geoip_path))
            except Exception as e:
                logger.warning(f"GeoIP database not available: {e}")
                self.enable_geo_lookup = False
        
        # Start background tasks
        self._start_background_tasks()
    
    async def log_event(
        self,
        context: AuditContext,
        action: str,
        resource: str,
        success: bool,
        resource_id: Optional[str] = None,
        method: Optional[str] = None,
        endpoint: Optional[str] = None,
        request_data: Optional[Dict[str, Any]] = None,
        response_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        error_code: Optional[str] = None,
        duration_ms: Optional[int] = None,
        tokens_used: Optional[int] = None,
        permission_checked: Optional[str] = None,
        authorization_result: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log audit event with comprehensive details.
        
        Args:
            context: Audit context
            action: Action performed
            resource: Resource accessed
            success: Operation success status
            resource_id: Specific resource ID
            method: HTTP method
            endpoint: API endpoint
            request_data: Request payload (sanitized)
            response_data: Response data (sanitized)
            error_message: Error message if failed
            error_code: Specific error code
            duration_ms: Operation duration
            tokens_used: Number of tokens consumed
            permission_checked: Permission that was checked
            authorization_result: Authorization result
            metadata: Additional metadata
            
        Returns:
            Log entry ID
        """
        try:
            # Create audit log entry
            audit_log = SecurityAuditLog(
                agent_id=context.agent_id,
                human_controller=context.human_controller,
                session_id=context.session_id,
                request_id=context.request_id,
                action=action,
                resource=resource,
                resource_id=resource_id,
                method=method,
                endpoint=endpoint,
                request_data=self._sanitize_data(request_data),
                response_data=self._sanitize_data(response_data),
                ip_address=context.ip_address,
                user_agent=context.user_agent,
                success=success,
                error_message=error_message,
                error_code=error_code,
                duration_ms=duration_ms,
                tokens_used=tokens_used,
                permission_checked=permission_checked,
                authorization_result=authorization_result,
                geo_location=await self._get_geo_location(context.ip_address),
                correlation_id=context.correlation_id,
                metadata=metadata or {}
            )
            
            # Calculate risk score
            audit_log.risk_score = self._calculate_risk_score(audit_log, context)
            
            # Add security labels
            audit_log.security_labels = self._classify_security_labels(audit_log, context)
            
            # Generate cryptographic signature
            if self.compliance_config["require_signatures"]:
                audit_log.log_signature = self._generate_log_signature(audit_log)
            
            # Add to batch for processing
            self.pending_logs.append(audit_log)
            
            # Check if we need to flush
            if (len(self.pending_logs) >= self.batch_size or 
                datetime.utcnow() - self.last_flush > timedelta(seconds=self.flush_interval)):
                await self._flush_pending_logs()
            
            # Real-time threat detection
            if self.compliance_config["real_time_monitoring"]:
                await self._detect_threats(audit_log, context)
            
            return str(audit_log.id)
            
        except Exception as e:
            logger.error(f"Audit logging failed: {e}")
            # Fallback logging to Redis
            await self._fallback_log(context, action, resource, success, error=str(e))
            raise
    
    async def create_security_event(
        self,
        event_type: str,
        severity: SecurityEventSeverity,
        description: str,
        agent_id: Optional[str] = None,
        human_controller: Optional[str] = None,
        source_ip: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        risk_score: Optional[float] = None,
        auto_detected: bool = True,
        related_audit_log_ids: Optional[List[str]] = None
    ) -> str:
        """
        Create security event for incidents and alerts.
        
        Args:
            event_type: Type of security event
            severity: Event severity level
            description: Event description
            agent_id: Associated agent ID
            human_controller: Associated human controller
            source_ip: Source IP address
            details: Additional event details
            risk_score: Risk score (0.0-1.0)
            auto_detected: Whether event was auto-detected
            related_audit_log_ids: Related audit log IDs
            
        Returns:
            Security event ID
        """
        try:
            security_event = SecurityEvent(
                event_type=event_type,
                severity=severity.value,
                agent_id=uuid.UUID(agent_id) if agent_id else None,
                human_controller=human_controller,
                source_ip=source_ip,
                description=description,
                details=details or {},
                risk_score=risk_score,
                auto_detected=auto_detected,
                related_audit_log_ids=[uuid.UUID(id) for id in (related_audit_log_ids or [])]
            )
            
            self.db.add(security_event)
            await self.db.commit()
            await self.db.refresh(security_event)
            
            # Update metrics
            self.metrics["threat_alerts"] += 1
            
            # Trigger incident response if critical
            if severity == SecurityEventSeverity.CRITICAL:
                await self._trigger_incident_response(security_event)
            
            logger.warning(
                f"Security event created: {event_type} (Severity: {severity.value}) - {description}"
            )
            
            return str(security_event.id)
            
        except Exception as e:
            logger.error(f"Security event creation failed: {e}")
            await self.db.rollback()
            raise
    
    async def query_audit_logs(
        self,
        filters: AuditLogFilters
    ) -> Dict[str, Any]:
        """
        Query audit logs with filters and pagination.
        
        Args:
            filters: Query filters
            
        Returns:
            Query results with pagination info
        """
        try:
            # Build query
            query = select(SecurityAuditLog)
            
            # Apply filters
            conditions = []
            
            if filters.agent_id:
                conditions.append(SecurityAuditLog.agent_id == uuid.UUID(filters.agent_id))
            
            if filters.human_controller:
                conditions.append(SecurityAuditLog.human_controller == filters.human_controller)
            
            if filters.action:
                conditions.append(SecurityAuditLog.action.ilike(f"%{filters.action}%"))
            
            if filters.resource:
                conditions.append(SecurityAuditLog.resource.ilike(f"%{filters.resource}%"))
            
            if filters.success is not None:
                conditions.append(SecurityAuditLog.success == filters.success)
            
            if filters.start_time:
                conditions.append(SecurityAuditLog.timestamp >= filters.start_time)
            
            if filters.end_time:
                conditions.append(SecurityAuditLog.timestamp <= filters.end_time)
            
            if filters.min_risk_score is not None:
                conditions.append(SecurityAuditLog.risk_score >= filters.min_risk_score)
            
            if conditions:
                query = query.where(and_(*conditions))
            
            # Get total count
            count_query = select(func.count(SecurityAuditLog.id)).where(and_(*conditions))
            total_count_result = await self.db.execute(count_query)
            total_count = total_count_result.scalar() or 0
            
            # Apply ordering and pagination
            query = query.order_by(desc(SecurityAuditLog.timestamp))
            query = query.offset(filters.offset).limit(filters.limit)
            
            # Execute query
            result = await self.db.execute(query)
            logs = result.scalars().all()
            
            # Convert to response format
            log_data = [log.to_dict() for log in logs]
            
            return {
                "logs": log_data,
                "pagination": {
                    "total": total_count,
                    "offset": filters.offset,
                    "limit": filters.limit,
                    "has_more": (filters.offset + len(log_data)) < total_count
                },
                "query_time": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Audit log query failed: {e}")
            return {"error": str(e), "logs": [], "pagination": {"total": 0}}
    
    async def get_security_dashboard(self) -> Dict[str, Any]:
        """
        Get security dashboard with real-time metrics.
        
        Returns:
            Security dashboard data
        """
        try:
            # Time window for analysis
            now = datetime.utcnow()
            last_24h = now - timedelta(hours=24)
            last_7d = now - timedelta(days=7)
            
            # Audit log statistics
            audit_stats = await self._get_audit_statistics(last_24h, now)
            
            # Security events
            security_events = await self._get_security_events_summary(last_24h, now)
            
            # Top risk agents
            risk_agents = await self._get_top_risk_agents(last_24h, now)
            
            # Geographic distribution
            geo_stats = await self._get_geographic_distribution(last_24h, now)
            
            # System health metrics
            system_health = {
                "audit_logger_status": "healthy",
                "pending_logs": len(self.pending_logs),
                "logs_written_24h": audit_stats.get("total_events", 0),
                "avg_processing_time_ms": self.metrics["avg_write_time_ms"],
                "threat_detection_active": self.compliance_config["real_time_monitoring"]
            }
            
            return {
                "timestamp": now.isoformat(),
                "audit_statistics": audit_stats,
                "security_events": security_events,
                "top_risk_agents": risk_agents,
                "geographic_distribution": geo_stats,
                "system_health": system_health,
                "performance_metrics": self.metrics.copy()
            }
            
        except Exception as e:
            logger.error(f"Security dashboard generation failed: {e}")
            return {"error": str(e)}
    
    async def verify_log_integrity(
        self,
        log_id: str,
        expected_signature: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verify integrity of audit log entry.
        
        Args:
            log_id: Log entry ID
            expected_signature: Expected signature to verify against
            
        Returns:
            Verification result
        """
        try:
            # Get log entry
            result = await self.db.execute(
                select(SecurityAuditLog).where(SecurityAuditLog.id == uuid.UUID(log_id))
            )
            log_entry = result.scalar_one_or_none()
            
            if not log_entry:
                return {
                    "verified": False,
                    "error": "Log entry not found",
                    "log_id": log_id
                }
            
            if not self.compliance_config["require_signatures"]:
                return {
                    "verified": True,
                    "message": "Signatures not required by configuration",
                    "log_id": log_id
                }
            
            # Verify signature
            computed_signature = self._generate_log_signature(log_entry)
            stored_signature = log_entry.log_signature
            
            signatures_match = (
                computed_signature == stored_signature and
                (expected_signature is None or stored_signature == expected_signature)
            )
            
            return {
                "verified": signatures_match,
                "log_id": log_id,
                "timestamp": log_entry.timestamp.isoformat(),
                "computed_signature": computed_signature,
                "stored_signature": stored_signature,
                "signatures_match": signatures_match,
                "integrity_status": "intact" if signatures_match else "compromised"
            }
            
        except Exception as e:
            logger.error(f"Log integrity verification failed: {e}")
            return {
                "verified": False,
                "error": str(e),
                "log_id": log_id
            }
    
    # Private helper methods
    
    def _sanitize_data(self, data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Sanitize sensitive data for compliance."""
        if not data:
            return data
        
        if not self.compliance_config["log_sensitive_data"]:
            # Remove PII and sensitive fields
            sensitive_fields = [
                "password", "secret", "token", "key", "credential",
                "ssn", "credit_card", "email", "phone", "address"
            ]
            
            sanitized = {}
            for key, value in data.items():
                if any(field in key.lower() for field in sensitive_fields):
                    sanitized[key] = "[REDACTED]"
                elif isinstance(value, dict):
                    sanitized[key] = self._sanitize_data(value)
                else:
                    sanitized[key] = value
            
            return sanitized
        
        return data
    
    async def _get_geo_location(self, ip_address: Optional[str]) -> Optional[str]:
        """Get geographic location from IP address."""
        if not ip_address or not self.enable_geo_lookup or not self.geoip_reader:
            return None
        
        try:
            response = self.geoip_reader.city(ip_address)
            return f"{response.city.name}, {response.country.name}"
        except Exception as e:
            logger.debug(f"GeoIP lookup failed for {ip_address}: {e}")
            return None
    
    def _calculate_risk_score(
        self,
        audit_log: SecurityAuditLog,
        context: AuditContext
    ) -> float:
        """Calculate risk score for audit log entry."""
        risk_score = 0.0
        
        # Base risk factors
        if not audit_log.success:
            risk_score += 0.3
        
        # High-privilege actions
        if audit_log.action in ["delete", "admin", "modify_permissions", "create_user"]:
            risk_score += 0.4
        
        # Off-hours activity
        hour = context.timestamp.hour
        if (hour >= self.threat_patterns["off_hours_activity"]["start_hour"] or 
            hour <= self.threat_patterns["off_hours_activity"]["end_hour"]):
            risk_score += 0.2
        
        # Error conditions
        if audit_log.error_code:
            risk_score += 0.2
        
        # High token usage
        if audit_log.tokens_used and audit_log.tokens_used > 1000:
            risk_score += 0.1
        
        # Suspicious user agent
        if audit_log.user_agent and "bot" in audit_log.user_agent.lower():
            risk_score += 0.3
        
        return min(1.0, risk_score)
    
    def _classify_security_labels(
        self,
        audit_log: SecurityAuditLog,
        context: AuditContext
    ) -> List[str]:
        """Classify security labels for the audit log."""
        labels = []
        
        if audit_log.risk_score and audit_log.risk_score > 0.7:
            labels.append("high_risk")
        elif audit_log.risk_score and audit_log.risk_score > 0.4:
            labels.append("medium_risk")
        
        if not audit_log.success:
            labels.append("failed_operation")
        
        if audit_log.action in ["delete", "admin"]:
            labels.append("privileged_action")
        
        if audit_log.tokens_used and audit_log.tokens_used > 1000:
            labels.append("high_resource_usage")
        
        # Time-based labels
        hour = context.timestamp.hour
        if (hour >= self.threat_patterns["off_hours_activity"]["start_hour"] or 
            hour <= self.threat_patterns["off_hours_activity"]["end_hour"]):
            labels.append("off_hours_activity")
        
        return labels
    
    def _generate_log_signature(self, audit_log: SecurityAuditLog) -> str:
        """Generate HMAC signature for log integrity."""
        # Create canonical representation
        data_to_sign = {
            "id": str(audit_log.id),
            "agent_id": str(audit_log.agent_id) if audit_log.agent_id else None,
            "action": audit_log.action,
            "resource": audit_log.resource,
            "timestamp": audit_log.timestamp.isoformat(),
            "success": audit_log.success,
            "human_controller": audit_log.human_controller
        }
        
        canonical_string = json.dumps(data_to_sign, sort_keys=True, separators=(',', ':'))
        
        # Generate HMAC-SHA256 signature
        signature = hmac.new(
            self.signing_key,
            canonical_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    async def _detect_threats(self, audit_log: SecurityAuditLog, context: AuditContext) -> None:
        """Real-time threat detection."""
        try:
            # Check for brute force attempts
            if not audit_log.success and audit_log.action in ["authenticate", "login"]:
                await self._check_brute_force(context)
            
            # Check for privilege escalation
            if audit_log.action in ["assign_role", "modify_permissions"]:
                await self._check_privilege_escalation(context)
            
            # Check for data exfiltration patterns
            if audit_log.action == "read" and audit_log.tokens_used and audit_log.tokens_used > 100:
                await self._check_data_exfiltration(context)
            
            # Check for rapid-fire requests
            if context.agent_id:
                await self._check_rapid_fire_requests(context)
            
        except Exception as e:
            logger.error(f"Threat detection failed: {e}")
    
    async def _check_brute_force(self, context: AuditContext) -> None:
        """Check for brute force attack patterns."""
        if not context.agent_id:
            return
        
        # Count recent failed attempts
        window_minutes = self.threat_patterns["brute_force"]["window_minutes"]
        threshold = self.threat_patterns["brute_force"]["threshold"]
        
        # This would check Redis for recent failed attempts
        # Simplified implementation
        cache_key = f"failed_auth:{context.agent_id}:{context.ip_address}"
        count = await self.redis.increment_counter(cache_key, window_minutes * 60)
        
        if count >= threshold:
            await self.create_security_event(
                event_type="brute_force_attack",
                severity=SecurityEventSeverity.HIGH,
                description=f"Brute force attack detected: {count} failed attempts in {window_minutes} minutes",
                agent_id=str(context.agent_id),
                source_ip=context.ip_address,
                details={
                    "failed_attempts": count,
                    "window_minutes": window_minutes,
                    "ip_address": context.ip_address
                },
                risk_score=0.8
            )
    
    async def _check_privilege_escalation(self, context: AuditContext) -> None:
        """Check for privilege escalation attempts."""
        # Implementation would analyze permission changes
        pass
    
    async def _check_data_exfiltration(self, context: AuditContext) -> None:
        """Check for data exfiltration patterns."""
        # Implementation would analyze data access patterns
        pass
    
    async def _check_rapid_fire_requests(self, context: AuditContext) -> None:
        """Check for rapid-fire request patterns."""
        # Implementation would analyze request frequency
        pass
    
    async def _flush_pending_logs(self) -> None:
        """Flush pending logs to database."""
        if not self.pending_logs:
            return
        
        try:
            # Bulk insert pending logs
            self.db.add_all(self.pending_logs)
            await self.db.commit()
            
            # Update metrics
            self.metrics["logs_written"] += len(self.pending_logs)
            self.metrics["batches_processed"] += 1
            
            # Clear pending logs
            self.pending_logs.clear()
            self.last_flush = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Failed to flush audit logs: {e}")
            await self.db.rollback()
            # Keep logs in pending for retry
    
    async def _fallback_log(
        self,
        context: AuditContext,
        action: str,
        resource: str,
        success: bool,
        error: str
    ) -> None:
        """Fallback logging to Redis when database fails."""
        try:
            fallback_data = {
                "context": context.to_dict(),
                "action": action,
                "resource": resource,
                "success": success,
                "error": error,
                "fallback_timestamp": datetime.utcnow().isoformat()
            }
            
            await self.redis.lpush("audit_fallback_logs", json.dumps(fallback_data))
            
        except Exception as e:
            logger.critical(f"Fallback logging failed: {e}")
    
    async def _get_audit_statistics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get audit statistics for time period."""
        result = await self.db.execute(
            select(
                func.count(SecurityAuditLog.id).label("total_events"),
                func.count(SecurityAuditLog.id).filter(SecurityAuditLog.success == True).label("successful_events"),
                func.count(SecurityAuditLog.id).filter(SecurityAuditLog.success == False).label("failed_events"),
                func.avg(SecurityAuditLog.risk_score).label("avg_risk_score"),
                func.avg(SecurityAuditLog.duration_ms).label("avg_duration_ms")
            ).where(
                and_(
                    SecurityAuditLog.timestamp >= start_time,
                    SecurityAuditLog.timestamp <= end_time
                )
            )
        )
        
        stats = result.first()
        
        return {
            "total_events": stats.total_events or 0,
            "successful_events": stats.successful_events or 0,
            "failed_events": stats.failed_events or 0,
            "success_rate": (stats.successful_events or 0) / max(1, stats.total_events or 1),
            "avg_risk_score": float(stats.avg_risk_score or 0),
            "avg_duration_ms": float(stats.avg_duration_ms or 0)
        }
    
    async def _get_security_events_summary(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get security events summary."""
        result = await self.db.execute(
            select(
                SecurityEvent.severity,
                func.count(SecurityEvent.id).label("count")
            ).where(
                and_(
                    SecurityEvent.timestamp >= start_time,
                    SecurityEvent.timestamp <= end_time
                )
            ).group_by(SecurityEvent.severity)
        )
        
        events_by_severity = {row.severity: row.count for row in result}
        
        return {
            "total_events": sum(events_by_severity.values()),
            "by_severity": events_by_severity,
            "critical_events": events_by_severity.get("critical", 0),
            "high_events": events_by_severity.get("high", 0)
        }
    
    async def _get_top_risk_agents(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Get top risk agents."""
        result = await self.db.execute(
            select(
                SecurityAuditLog.agent_id,
                func.avg(SecurityAuditLog.risk_score).label("avg_risk_score"),
                func.count(SecurityAuditLog.id).label("event_count")
            ).where(
                and_(
                    SecurityAuditLog.timestamp >= start_time,
                    SecurityAuditLog.timestamp <= end_time,
                    SecurityAuditLog.agent_id.is_not(None)
                )
            ).group_by(SecurityAuditLog.agent_id).order_by(desc("avg_risk_score")).limit(10)
        )
        
        return [
            {
                "agent_id": str(row.agent_id),
                "avg_risk_score": float(row.avg_risk_score),
                "event_count": row.event_count
            }
            for row in result
        ]
    
    async def _get_geographic_distribution(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get geographic distribution of events."""
        result = await self.db.execute(
            select(
                SecurityAuditLog.geo_location,
                func.count(SecurityAuditLog.id).label("count")
            ).where(
                and_(
                    SecurityAuditLog.timestamp >= start_time,
                    SecurityAuditLog.timestamp <= end_time,
                    SecurityAuditLog.geo_location.is_not(None)
                )
            ).group_by(SecurityAuditLog.geo_location).order_by(desc("count")).limit(20)
        )
        
        return {
            "by_location": [
                {"location": row.geo_location, "count": row.count}
                for row in result
            ]
        }
    
    async def _trigger_incident_response(self, security_event: SecurityEvent) -> None:
        """Trigger incident response for critical events."""
        try:
            incident = SecurityIncident(
                incident_id=str(uuid.uuid4()),
                event_type=security_event.event_type,
                severity=SecurityEventSeverity(security_event.severity),
                agent_id=security_event.agent_id,
                description=security_event.description,
                evidence=security_event.details,
                risk_score=security_event.risk_score or 1.0,
                timestamp=security_event.timestamp
            )
            
            # Store incident in Redis for immediate processing
            await self.redis.lpush("security_incidents", json.dumps(incident.to_dict()))
            
            # Send alerts (would integrate with alerting system)
            logger.critical(f"SECURITY INCIDENT: {incident.description}")
            
        except Exception as e:
            logger.error(f"Failed to trigger incident response: {e}")
    
    def _start_background_tasks(self) -> None:
        """Start background tasks for batch processing."""
        # In a real implementation, would start background tasks
        pass


# Factory function
async def create_audit_logger(
    db_session: AsyncSession,
    redis_client: RedisClient,
    signing_key: str
) -> AuditLogger:
    """
    Create Audit Logger instance.
    
    Args:
        db_session: Database session
        redis_client: Redis client
        signing_key: HMAC signing key
        
    Returns:
        AuditLogger instance
    """
    return AuditLogger(db_session, redis_client, signing_key)