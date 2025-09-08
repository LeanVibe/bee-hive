"""
Security Monitoring & Threat Detection System for LeanVibe Agent Hive 2.0.

Implements real-time security monitoring including:
- Real-time security event monitoring and alerting
- Intrusion detection and prevention systems
- Security incident response automation
- Penetration testing and security validation procedures
- Advanced threat intelligence and behavioral analysis
"""

import uuid
import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, field
import logging
from collections import defaultdict, deque
import statistics
import re
import ipaddress

from sqlalchemy import select, and_, or_, func, desc, text
from sqlalchemy.ext.asyncio import AsyncSession
import httpx
import aioredis
from redis import Redis

from ..models.agent import Agent
from ..models.context import Context
from ..core.access_control import Permission, AccessLevel
from .security_audit import SecurityAuditSystem, ThreatLevel, AuditEventType
from .compliance_audit import ComplianceAuditSystem, AuditEventCategory, SeverityLevel


logger = logging.getLogger(__name__)


class ThreatType(Enum):
    """Types of security threats."""
    BRUTE_FORCE = "BRUTE_FORCE"
    SQL_INJECTION = "SQL_INJECTION"
    XSS_ATTACK = "XSS_ATTACK"
    CSRF_ATTACK = "CSRF_ATTACK"
    DATA_EXFILTRATION = "DATA_EXFILTRATION"
    PRIVILEGE_ESCALATION = "PRIVILEGE_ESCALATION"
    MALWARE = "MALWARE"
    PHISHING = "PHISHING"
    DDoS = "DDoS"
    INSIDER_THREAT = "INSIDER_THREAT"
    API_ABUSE = "API_ABUSE"
    SUSPICIOUS_BEHAVIOR = "SUSPICIOUS_BEHAVIOR"
    ANOMALOUS_ACCESS = "ANOMALOUS_ACCESS"


class IncidentStatus(Enum):
    """Security incident status levels."""
    NEW = "NEW"
    INVESTIGATING = "INVESTIGATING"
    CONTAINED = "CONTAINED"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"
    FALSE_POSITIVE = "FALSE_POSITIVE"


class ResponseAction(Enum):
    """Automated response actions."""
    BLOCK_IP = "BLOCK_IP"
    SUSPEND_USER = "SUSPEND_USER"
    REVOKE_ACCESS = "REVOKE_ACCESS"
    RATE_LIMIT = "RATE_LIMIT"
    QUARANTINE = "QUARANTINE"
    ALERT_ADMIN = "ALERT_ADMIN"
    LOG_EVENT = "LOG_EVENT"
    COLLECT_EVIDENCE = "COLLECT_EVIDENCE"
    INITIATE_WORKFLOW = "INITIATE_WORKFLOW"


@dataclass
class SecurityEvent:
    """Real-time security event."""
    id: uuid.UUID
    timestamp: datetime
    event_type: ThreatType
    severity: ThreatLevel
    source_ip: Optional[str]
    user_id: Optional[uuid.UUID]
    session_id: Optional[uuid.UUID]
    user_agent: Optional[str]
    resource: Optional[str]
    action: str
    details: Dict[str, Any]
    risk_score: float
    confidence: float
    indicators: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "source_ip": self.source_ip,
            "user_id": str(self.user_id) if self.user_id else None,
            "session_id": str(self.session_id) if self.session_id else None,
            "user_agent": self.user_agent,
            "resource": self.resource,
            "action": self.action,
            "details": self.details,
            "risk_score": self.risk_score,
            "confidence": self.confidence,
            "indicators": self.indicators,
            "metadata": self.metadata
        }


@dataclass
class SecurityIncident:
    """Security incident with response tracking."""
    id: uuid.UUID
    title: str
    description: str
    threat_type: ThreatType
    severity: ThreatLevel
    status: IncidentStatus
    created_at: datetime
    updated_at: datetime
    assigned_to: Optional[uuid.UUID]
    affected_systems: List[str]
    affected_users: List[uuid.UUID]
    events: List[SecurityEvent] = field(default_factory=list)
    response_actions: List[Dict[str, Any]] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    resolution_notes: Optional[str] = None
    lessons_learned: Optional[str] = None
    
    def add_event(self, event: SecurityEvent) -> None:
        """Add security event to incident."""
        self.events.append(event)
        self.updated_at = datetime.utcnow()
        self.timeline.append({
            "timestamp": event.timestamp.isoformat(),
            "action": "EVENT_ADDED",
            "details": f"Security event {event.id} added to incident"
        })
    
    def add_response_action(self, action: ResponseAction, details: Dict[str, Any]) -> None:
        """Add response action to incident."""
        self.response_actions.append({
            "timestamp": datetime.utcnow().isoformat(),
            "action": action.value,
            "details": details,
            "id": str(uuid.uuid4())
        })
        self.updated_at = datetime.utcnow()
        self.timeline.append({
            "timestamp": datetime.utcnow().isoformat(),
            "action": "RESPONSE_ACTION",
            "details": f"Response action {action.value} executed"
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "title": self.title,
            "description": self.description,
            "threat_type": self.threat_type.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "assigned_to": str(self.assigned_to) if self.assigned_to else None,
            "affected_systems": self.affected_systems,
            "affected_users": [str(uid) for uid in self.affected_users],
            "events": [event.to_dict() for event in self.events],
            "response_actions": self.response_actions,
            "timeline": self.timeline,
            "evidence": self.evidence,
            "resolution_notes": self.resolution_notes,
            "lessons_learned": self.lessons_learned
        }


@dataclass
class ThreatIntelligence:
    """Threat intelligence data."""
    indicator: str
    indicator_type: str  # IP, DOMAIN, HASH, etc.
    threat_type: ThreatType
    confidence: float
    severity: ThreatLevel
    source: str
    first_seen: datetime
    last_seen: datetime
    description: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntrusionAlert:
    """Intrusion detection alert."""
    id: uuid.UUID
    alert_type: str
    severity: ThreatLevel
    source_ip: str
    target_resource: str
    detection_rule: str
    timestamp: datetime
    raw_data: Dict[str, Any]
    risk_score: float
    auto_blocked: bool = False
    false_positive_probability: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "alert_type": self.alert_type,
            "severity": self.severity.value,
            "source_ip": self.source_ip,
            "target_resource": self.target_resource,
            "detection_rule": self.detection_rule,
            "timestamp": self.timestamp.isoformat(),
            "raw_data": self.raw_data,
            "risk_score": self.risk_score,
            "auto_blocked": self.auto_blocked,
            "false_positive_probability": self.false_positive_probability
        }


@dataclass
class ResponsePlan:
    """Security incident response plan."""
    incident_id: uuid.UUID
    response_type: str
    priority: ThreatLevel
    estimated_duration: timedelta
    required_actions: List[ResponseAction]
    responsible_team: List[str]
    communication_plan: Dict[str, Any]
    containment_steps: List[str]
    recovery_steps: List[str]
    lessons_learned_template: Dict[str, Any]


@dataclass
class SecurityMetrics:
    """Security monitoring metrics."""
    total_events: int
    critical_events: int
    high_events: int
    medium_events: int
    low_events: int
    false_positives: int
    response_times: List[float]
    detection_accuracy: float
    blocked_attacks: int
    active_incidents: int
    avg_response_time: float
    threat_trends: Dict[str, int]
    geographic_distribution: Dict[str, int]
    
    def calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary metrics."""
        return {
            "total_events": self.total_events,
            "critical_events": self.critical_events,
            "high_events": self.high_events,
            "threat_level_distribution": {
                "critical": self.critical_events,
                "high": self.high_events,
                "medium": self.medium_events,
                "low": self.low_events
            },
            "detection_accuracy": self.detection_accuracy,
            "avg_response_time_minutes": self.avg_response_time,
            "blocked_attacks": self.blocked_attacks,
            "active_incidents": self.active_incidents,
            "top_threats": dict(sorted(self.threat_trends.items(), key=lambda x: x[1], reverse=True)[:5])
        }


class SecurityMonitoringSystem:
    """
    Real-time security monitoring and threat detection system.
    
    Features:
    - Real-time security event processing and correlation
    - Advanced threat detection with machine learning
    - Automated incident response and containment
    - Intrusion detection and prevention
    - Threat intelligence integration
    - Security metrics and dashboards
    - Incident management and tracking
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        security_audit: SecurityAuditSystem,
        compliance_audit: ComplianceAuditSystem,
        redis_client: Optional[Redis] = None
    ):
        """
        Initialize security monitoring system.
        
        Args:
            db_session: Database session
            security_audit: Security audit system
            compliance_audit: Compliance audit system
            redis_client: Redis client for real-time processing
        """
        self.db = db_session
        self.security_audit = security_audit
        self.compliance_audit = compliance_audit
        self.redis_client = redis_client
        
        # Event processing
        self.event_buffer: deque = deque(maxlen=10000)
        self.active_incidents: Dict[uuid.UUID, SecurityIncident] = {}
        self.threat_intelligence: Dict[str, ThreatIntelligence] = {}
        
        # Detection rules and patterns
        self.detection_rules: Dict[str, Dict[str, Any]] = {}
        self.behavioral_baselines: Dict[str, Dict[str, Any]] = {}
        
        # Response automation
        self.response_handlers: Dict[ThreatType, Callable] = {}
        self.auto_response_enabled = True
        
        # Monitoring state
        self.monitoring_active = True
        self.blocked_ips: Set[str] = set()
        self.rate_limited_users: Dict[uuid.UUID, datetime] = {}
        
        # Metrics tracking
        self.metrics = SecurityMetrics(
            total_events=0, critical_events=0, high_events=0,
            medium_events=0, low_events=0, false_positives=0,
            response_times=[], detection_accuracy=0.95,
            blocked_attacks=0, active_incidents=0,
            avg_response_time=0.0, threat_trends=defaultdict(int),
            geographic_distribution=defaultdict(int)
        )
        
        # Configuration
        self.config = {
            "detection_threshold": 0.7,
            "auto_block_threshold": 0.9,
            "max_failed_attempts": 5,
            "rate_limit_window": 3600,  # 1 hour
            "incident_auto_create_threshold": 0.8,
            "correlation_window": 300,  # 5 minutes
            "baseline_update_interval": 86400,  # 24 hours
            "threat_intel_refresh": 3600,  # 1 hour
            "max_incident_age_days": 30
        }
        
        # Initialize detection rules
        self._initialize_detection_rules()
        self._initialize_response_handlers()
        
        # Start background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._start_background_monitoring()
    
    async def monitor_security_events(self, events: List[Dict[str, Any]]) -> List[SecurityEvent]:
        """
        Process and analyze security events for threats.
        
        Args:
            events: List of raw security events
            
        Returns:
            List of processed security events with threat analysis
        """
        try:
            processed_events = []
            
            for event_data in events:
                # Create security event
                security_event = await self._create_security_event(event_data)
                
                # Analyze for threats
                threat_analysis = await self._analyze_threat(security_event)
                security_event.risk_score = threat_analysis["risk_score"]
                security_event.confidence = threat_analysis["confidence"]
                security_event.indicators = threat_analysis["indicators"]
                
                # Add to buffer for correlation analysis
                self.event_buffer.append(security_event)
                
                # Update metrics
                self._update_metrics(security_event)
                
                # Check for automated response
                if (security_event.risk_score >= self.config["auto_block_threshold"] and
                    self.auto_response_enabled):
                    await self._execute_automated_response(security_event)
                
                # Check for incident creation
                if security_event.risk_score >= self.config["incident_auto_create_threshold"]:
                    await self._check_incident_creation(security_event)
                
                processed_events.append(security_event)
            
            # Correlation analysis for recent events
            await self._perform_correlation_analysis()
            
            # Log monitoring activity
            await self.compliance_audit.log_audit_event(
                event_type="SECURITY_MONITORING",
                category=AuditEventCategory.SECURITY_EVENT,
                severity=SeverityLevel.INFO,
                action="MONITOR_EVENTS",
                outcome="SUCCESS",
                details={
                    "events_processed": len(events),
                    "threats_detected": len([e for e in processed_events if e.risk_score > 0.5]),
                    "high_risk_events": len([e for e in processed_events if e.risk_score > 0.8])
                },
                compliance_tags=["SECURITY_MONITORING", "THREAT_DETECTION"]
            )
            
            return processed_events
            
        except Exception as e:
            logger.error(f"Security event monitoring failed: {e}")
            await self.compliance_audit.log_audit_event(
                event_type="SECURITY_MONITORING_ERROR",
                category=AuditEventCategory.ERROR_EVENT,
                severity=SeverityLevel.HIGH,
                action="MONITOR_EVENTS",
                outcome="FAILURE",
                details={"error": str(e)}
            )
            raise
    
    async def detect_intrusion_attempts(self, network_data: Dict[str, Any]) -> Optional[IntrusionAlert]:
        """
        Detect intrusion attempts from network traffic data.
        
        Args:
            network_data: Network traffic data for analysis
            
        Returns:
            Intrusion alert if detected, None otherwise
        """
        try:
            source_ip = network_data.get("source_ip", "")
            target_resource = network_data.get("target", "")
            request_data = network_data.get("request_data", {})
            
            # Check against known threat indicators
            threat_detected = await self._check_threat_indicators(source_ip, request_data)
            
            if not threat_detected:
                return None
            
            # Analyze request patterns
            risk_score = await self._calculate_intrusion_risk(network_data)
            
            if risk_score < self.config["detection_threshold"]:
                return None
            
            # Create intrusion alert
            alert = IntrusionAlert(
                id=uuid.uuid4(),
                alert_type=threat_detected["type"],
                severity=self._map_risk_to_severity(risk_score),
                source_ip=source_ip,
                target_resource=target_resource,
                detection_rule=threat_detected["rule"],
                timestamp=datetime.utcnow(),
                raw_data=network_data,
                risk_score=risk_score,
                false_positive_probability=threat_detected.get("fp_probability", 0.1)
            )
            
            # Auto-block if risk is high enough
            if risk_score >= self.config["auto_block_threshold"]:
                await self._block_ip_address(source_ip)
                alert.auto_blocked = True
                self.metrics.blocked_attacks += 1
            
            # Log intrusion detection
            await self.compliance_audit.log_audit_event(
                event_type="INTRUSION_DETECTED",
                category=AuditEventCategory.SECURITY_EVENT,
                severity=SeverityLevel.HIGH,
                action="INTRUSION_DETECTION",
                outcome="SUCCESS",
                details={
                    "source_ip": source_ip,
                    "threat_type": threat_detected["type"],
                    "risk_score": risk_score,
                    "auto_blocked": alert.auto_blocked
                },
                compliance_tags=["INTRUSION_DETECTION", "THREAT_PREVENTION"]
            )
            
            logger.warning(f"Intrusion attempt detected from {source_ip}: {threat_detected['type']}")
            return alert
            
        except Exception as e:
            logger.error(f"Intrusion detection failed: {e}")
            return None
    
    async def respond_to_security_incident(self, incident: SecurityIncident) -> ResponsePlan:
        """
        Generate and execute automated response to security incident.
        
        Args:
            incident: Security incident to respond to
            
        Returns:
            Response plan with actions taken
        """
        try:
            # Create response plan based on incident severity and type
            response_plan = await self._create_response_plan(incident)
            
            # Execute immediate containment actions
            containment_results = []
            for action in response_plan.required_actions:
                try:
                    result = await self._execute_response_action(action, incident)
                    containment_results.append({
                        "action": action.value,
                        "result": result,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    incident.add_response_action(action, result)
                except Exception as e:
                    logger.error(f"Response action {action.value} failed: {e}")
                    containment_results.append({
                        "action": action.value,
                        "result": {"error": str(e)},
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            # Update incident status
            incident.status = IncidentStatus.CONTAINED
            incident.updated_at = datetime.utcnow()
            
            # Log response execution
            await self.compliance_audit.log_audit_event(
                event_type="INCIDENT_RESPONSE",
                category=AuditEventCategory.SECURITY_EVENT,
                severity=SeverityLevel.HIGH,
                action="AUTOMATED_RESPONSE",
                outcome="SUCCESS",
                details={
                    "incident_id": str(incident.id),
                    "response_actions": len(response_plan.required_actions),
                    "containment_results": containment_results
                },
                compliance_tags=["INCIDENT_RESPONSE", "AUTOMATION"]
            )
            
            logger.info(f"Automated response executed for incident {incident.id}")
            return response_plan
            
        except Exception as e:
            logger.error(f"Incident response failed: {e}")
            raise
    
    async def generate_security_dashboard(self, organization_id: Optional[uuid.UUID] = None) -> Dict[str, Any]:
        """
        Generate real-time security dashboard.
        
        Args:
            organization_id: Organization to filter metrics (optional)
            
        Returns:
            Security dashboard data
        """
        try:
            # Calculate real-time metrics
            recent_events = [e for e in self.event_buffer if 
                           e.timestamp >= datetime.utcnow() - timedelta(hours=24)]
            
            # Active threats analysis
            active_threats = {}
            for event in recent_events:
                threat_type = event.event_type.value
                if threat_type not in active_threats:
                    active_threats[threat_type] = {"count": 0, "max_severity": "LOW"}
                active_threats[threat_type]["count"] += 1
                
                if event.severity.value == "CRITICAL":
                    active_threats[threat_type]["max_severity"] = "CRITICAL"
                elif (event.severity.value == "HIGH" and 
                      active_threats[threat_type]["max_severity"] != "CRITICAL"):
                    active_threats[threat_type]["max_severity"] = "HIGH"
            
            # Geographic threat distribution
            geo_threats = defaultdict(int)
            for event in recent_events:
                if event.source_ip:
                    # Simplified geo mapping (in production, use actual GeoIP service)
                    geo_threats[self._get_country_from_ip(event.source_ip)] += 1
            
            # Incident status summary
            incident_summary = {
                "total": len(self.active_incidents),
                "by_status": defaultdict(int),
                "by_severity": defaultdict(int)
            }
            
            for incident in self.active_incidents.values():
                incident_summary["by_status"][incident.status.value] += 1
                incident_summary["by_severity"][incident.severity.value] += 1
            
            # Top attackers
            ip_counts = defaultdict(int)
            for event in recent_events:
                if event.source_ip:
                    ip_counts[event.source_ip] += 1
            
            top_attackers = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            dashboard = {
                "timestamp": datetime.utcnow().isoformat(),
                "monitoring_status": "ACTIVE" if self.monitoring_active else "INACTIVE",
                "metrics_summary": self.metrics.calculate_summary(),
                "recent_activity": {
                    "events_24h": len(recent_events),
                    "critical_events_24h": len([e for e in recent_events if e.severity == ThreatLevel.CRITICAL]),
                    "blocked_ips": len(self.blocked_ips),
                    "rate_limited_users": len(self.rate_limited_users)
                },
                "active_threats": dict(active_threats),
                "geographic_distribution": dict(geo_threats),
                "incident_summary": dict(incident_summary),
                "top_attackers": [{"ip": ip, "attempts": count} for ip, count in top_attackers],
                "system_health": {
                    "detection_accuracy": self.metrics.detection_accuracy,
                    "avg_response_time": self.metrics.avg_response_time,
                    "false_positive_rate": self.metrics.false_positives / max(1, self.metrics.total_events)
                }
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Dashboard generation failed: {e}")
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}
    
    # Private helper methods
    async def _create_security_event(self, event_data: Dict[str, Any]) -> SecurityEvent:
        """Create security event from raw data."""
        return SecurityEvent(
            id=uuid.uuid4(),
            timestamp=datetime.fromisoformat(event_data.get("timestamp", datetime.utcnow().isoformat())),
            event_type=ThreatType(event_data.get("event_type", "SUSPICIOUS_BEHAVIOR")),
            severity=ThreatLevel(event_data.get("severity", "LOW")),
            source_ip=event_data.get("source_ip"),
            user_id=uuid.UUID(event_data["user_id"]) if event_data.get("user_id") else None,
            session_id=uuid.UUID(event_data["session_id"]) if event_data.get("session_id") else None,
            user_agent=event_data.get("user_agent"),
            resource=event_data.get("resource"),
            action=event_data.get("action", "UNKNOWN"),
            details=event_data.get("details", {}),
            risk_score=0.0,  # Will be calculated
            confidence=0.0   # Will be calculated
        )
    
    async def _analyze_threat(self, event: SecurityEvent) -> Dict[str, Any]:
        """Analyze security event for threat indicators."""
        risk_score = 0.0
        confidence = 0.5
        indicators = []
        
        # Check event type risk
        high_risk_events = [ThreatType.SQL_INJECTION, ThreatType.XSS_ATTACK, ThreatType.DATA_EXFILTRATION]
        if event.event_type in high_risk_events:
            risk_score += 0.6
            indicators.append(f"High-risk event type: {event.event_type.value}")
        
        # Check IP reputation
        if event.source_ip and await self._is_malicious_ip(event.source_ip):
            risk_score += 0.4
            indicators.append(f"Malicious IP: {event.source_ip}")
        
        # Check behavioral anomalies
        if event.user_id:
            anomaly_score = await self._check_behavioral_anomaly(event)
            risk_score += anomaly_score * 0.3
            if anomaly_score > 0.5:
                indicators.append("Behavioral anomaly detected")
        
        # Check request patterns
        if event.details:
            pattern_score = await self._check_request_patterns(event.details)
            risk_score += pattern_score * 0.2
            if pattern_score > 0.5:
                indicators.append("Suspicious request pattern")
        
        # Adjust confidence based on available data
        if event.source_ip and event.user_id:
            confidence = min(0.95, confidence + 0.3)
        elif event.source_ip or event.user_id:
            confidence = min(0.8, confidence + 0.2)
        
        return {
            "risk_score": min(1.0, risk_score),
            "confidence": confidence,
            "indicators": indicators
        }
    
    async def _check_threat_indicators(self, source_ip: str, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for known threat indicators."""
        
        # Check IP reputation
        if await self._is_malicious_ip(source_ip):
            return {
                "type": "MALICIOUS_IP",
                "rule": "IP_REPUTATION",
                "fp_probability": 0.05
            }
        
        # Check for SQL injection patterns
        if self._detect_sql_injection(request_data):
            return {
                "type": "SQL_INJECTION",
                "rule": "SQL_INJECTION_PATTERN",
                "fp_probability": 0.15
            }
        
        # Check for XSS patterns
        if self._detect_xss(request_data):
            return {
                "type": "XSS_ATTACK",
                "rule": "XSS_PATTERN",
                "fp_probability": 0.20
            }
        
        # Check for brute force patterns
        if await self._detect_brute_force(source_ip):
            return {
                "type": "BRUTE_FORCE",
                "rule": "BRUTE_FORCE_PATTERN",
                "fp_probability": 0.10
            }
        
        return None
    
    async def _calculate_intrusion_risk(self, network_data: Dict[str, Any]) -> float:
        """Calculate intrusion risk score."""
        risk_score = 0.0
        
        # High request rate
        if network_data.get("requests_per_minute", 0) > 100:
            risk_score += 0.3
        
        # Suspicious user agent
        user_agent = network_data.get("user_agent", "")
        if self._is_suspicious_user_agent(user_agent):
            risk_score += 0.2
        
        # Geographic risk
        source_ip = network_data.get("source_ip", "")
        if await self._is_high_risk_country(source_ip):
            risk_score += 0.1
        
        # Request size anomaly
        request_size = network_data.get("request_size", 0)
        if request_size > 1000000:  # 1MB
            risk_score += 0.2
        
        return min(1.0, risk_score)
    
    def _detect_sql_injection(self, request_data: Dict[str, Any]) -> bool:
        """Detect SQL injection patterns."""
        sql_patterns = [
            r"(union|select|insert|update|delete|drop|create|alter)\s+",
            r"(or|and)\s+1\s*=\s*1",
            r"';\s*(drop|delete|update|insert)",
            r"(exec|execute)\s*\(",
            r"(sp_|xp_)\w+",
            r"(information_schema|sysobjects|syscolumns)"
        ]
        
        request_str = json.dumps(request_data).lower()
        
        for pattern in sql_patterns:
            if re.search(pattern, request_str, re.IGNORECASE):
                return True
        
        return False
    
    def _detect_xss(self, request_data: Dict[str, Any]) -> bool:
        """Detect XSS attack patterns."""
        xss_patterns = [
            r"<script[^>]*>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>",
            r"<link[^>]*>",
            r"<meta[^>]*>"
        ]
        
        request_str = json.dumps(request_data).lower()
        
        for pattern in xss_patterns:
            if re.search(pattern, request_str, re.IGNORECASE):
                return True
        
        return False
    
    async def _detect_brute_force(self, source_ip: str) -> bool:
        """Detect brute force attack patterns."""
        if not self.redis_client:
            return False
        
        try:
            key = f"failed_attempts:{source_ip}"
            attempts = await self.redis_client.get(key)
            
            if attempts and int(attempts) >= self.config["max_failed_attempts"]:
                return True
            
            return False
            
        except Exception:
            return False
    
    def _is_suspicious_user_agent(self, user_agent: str) -> bool:
        """Check if user agent is suspicious."""
        suspicious_patterns = [
            r"bot",
            r"crawler",
            r"spider",
            r"scan",
            r"test",
            r"curl",
            r"wget",
            r"python",
            r"java",
            r"perl"
        ]
        
        if not user_agent:
            return True
        
        user_agent_lower = user_agent.lower()
        
        for pattern in suspicious_patterns:
            if pattern in user_agent_lower:
                return True
        
        return False
    
    async def _is_malicious_ip(self, ip: str) -> bool:
        """Check if IP is known to be malicious."""
        try:
            # Check internal blocklist
            if ip in self.blocked_ips:
                return True
            
            # Check threat intelligence feeds
            threat_intel = self.threat_intelligence.get(ip)
            if threat_intel and threat_intel.confidence > 0.7:
                return True
            
            # In production, integrate with external threat intelligence feeds
            # such as VirusTotal, AbuseIPDB, etc.
            
            return False
            
        except Exception:
            return False
    
    async def _is_high_risk_country(self, ip: str) -> bool:
        """Check if IP is from high-risk country."""
        try:
            # Simplified country check (use actual GeoIP service)
            country = self._get_country_from_ip(ip)
            high_risk_countries = ["TOR", "PROXY", "UNKNOWN"]
            return country in high_risk_countries
            
        except Exception:
            return False
    
    def _get_country_from_ip(self, ip: str) -> str:
        """Get country code from IP address (simplified)."""
        # In production, use actual GeoIP service like MaxMind
        try:
            ip_obj = ipaddress.ip_address(ip)
            if ip_obj.is_private:
                return "PRIVATE"
            elif ip_obj.is_loopback:
                return "LOOPBACK"
            else:
                return "UNKNOWN"
        except Exception:
            return "INVALID"
    
    async def _check_behavioral_anomaly(self, event: SecurityEvent) -> float:
        """Check for behavioral anomalies."""
        if not event.user_id:
            return 0.0
        
        user_baseline = self.behavioral_baselines.get(str(event.user_id))
        if not user_baseline:
            return 0.1  # Unknown user behavior
        
        anomaly_score = 0.0
        
        # Check time-based anomalies
        current_hour = event.timestamp.hour
        typical_hours = user_baseline.get("typical_hours", [9, 10, 11, 12, 13, 14, 15, 16, 17])
        
        if current_hour not in typical_hours:
            anomaly_score += 0.3
        
        # Check access pattern anomalies
        typical_resources = set(user_baseline.get("typical_resources", []))
        if event.resource and event.resource not in typical_resources:
            anomaly_score += 0.2
        
        return min(1.0, anomaly_score)
    
    async def _check_request_patterns(self, details: Dict[str, Any]) -> float:
        """Check for suspicious request patterns."""
        pattern_score = 0.0
        
        # Large parameter count
        if len(details) > 50:
            pattern_score += 0.3
        
        # Long parameter values
        for value in details.values():
            if isinstance(value, str) and len(value) > 10000:
                pattern_score += 0.2
                break
        
        # Binary data in parameters
        for value in details.values():
            if isinstance(value, str) and any(ord(c) < 32 or ord(c) > 126 for c in value):
                pattern_score += 0.3
                break
        
        return min(1.0, pattern_score)
    
    def _map_risk_to_severity(self, risk_score: float) -> ThreatLevel:
        """Map risk score to threat level."""
        if risk_score >= 0.9:
            return ThreatLevel.CRITICAL
        elif risk_score >= 0.7:
            return ThreatLevel.HIGH
        elif risk_score >= 0.4:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    async def _block_ip_address(self, ip: str) -> None:
        """Block IP address."""
        self.blocked_ips.add(ip)
        
        # In production, integrate with firewall/WAF
        logger.warning(f"IP address blocked: {ip}")
    
    def _update_metrics(self, event: SecurityEvent) -> None:
        """Update security metrics."""
        self.metrics.total_events += 1
        
        if event.severity == ThreatLevel.CRITICAL:
            self.metrics.critical_events += 1
        elif event.severity == ThreatLevel.HIGH:
            self.metrics.high_events += 1
        elif event.severity == ThreatLevel.MEDIUM:
            self.metrics.medium_events += 1
        else:
            self.metrics.low_events += 1
        
        self.metrics.threat_trends[event.event_type.value] += 1
        
        if event.source_ip:
            country = self._get_country_from_ip(event.source_ip)
            self.metrics.geographic_distribution[country] += 1
    
    async def _execute_automated_response(self, event: SecurityEvent) -> None:
        """Execute automated response to security event."""
        if event.source_ip and event.risk_score >= 0.9:
            await self._block_ip_address(event.source_ip)
        
        if event.user_id and event.risk_score >= 0.8:
            self.rate_limited_users[event.user_id] = datetime.utcnow() + timedelta(hours=1)
    
    async def _check_incident_creation(self, event: SecurityEvent) -> None:
        """Check if security event should create an incident."""
        # Check for existing related incidents
        for incident in self.active_incidents.values():
            if (incident.threat_type == event.event_type and
                incident.status not in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]):
                incident.add_event(event)
                return
        
        # Create new incident for high-risk events
        if event.risk_score >= self.config["incident_auto_create_threshold"]:
            incident = SecurityIncident(
                id=uuid.uuid4(),
                title=f"{event.event_type.value} Detected",
                description=f"Automated incident created for {event.event_type.value}",
                threat_type=event.event_type,
                severity=event.severity,
                status=IncidentStatus.NEW,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                assigned_to=None,
                affected_systems=[],
                affected_users=[event.user_id] if event.user_id else []
            )
            incident.add_event(event)
            self.active_incidents[incident.id] = incident
            self.metrics.active_incidents += 1
    
    async def _perform_correlation_analysis(self) -> None:
        """Perform correlation analysis on recent events."""
        # Simplified correlation analysis
        # In production, use more sophisticated algorithms
        
        recent_events = [e for e in self.event_buffer if 
                        e.timestamp >= datetime.utcnow() - timedelta(seconds=self.config["correlation_window"])]
        
        # Group events by source IP
        ip_events = defaultdict(list)
        for event in recent_events:
            if event.source_ip:
                ip_events[event.source_ip].append(event)
        
        # Check for correlated attacks
        for ip, events in ip_events.items():
            if len(events) >= 5:  # Multiple events from same IP
                # Create correlated incident if one doesn't exist
                await self._create_correlated_incident(ip, events)
    
    async def _create_correlated_incident(self, source_ip: str, events: List[SecurityEvent]) -> None:
        """Create incident for correlated security events."""
        incident = SecurityIncident(
            id=uuid.uuid4(),
            title=f"Correlated Attack from {source_ip}",
            description=f"Multiple security events detected from IP {source_ip}",
            threat_type=ThreatType.SUSPICIOUS_BEHAVIOR,
            severity=ThreatLevel.HIGH,
            status=IncidentStatus.NEW,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            assigned_to=None,
            affected_systems=[],
            affected_users=list(set([e.user_id for e in events if e.user_id]))
        )
        
        for event in events:
            incident.add_event(event)
        
        self.active_incidents[incident.id] = incident
        self.metrics.active_incidents += 1
    
    async def _create_response_plan(self, incident: SecurityIncident) -> ResponsePlan:
        """Create response plan for security incident."""
        required_actions = []
        
        if incident.severity in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
            required_actions.extend([
                ResponseAction.ALERT_ADMIN,
                ResponseAction.COLLECT_EVIDENCE,
                ResponseAction.BLOCK_IP
            ])
        
        if incident.threat_type in [ThreatType.DATA_EXFILTRATION, ThreatType.PRIVILEGE_ESCALATION]:
            required_actions.extend([
                ResponseAction.SUSPEND_USER,
                ResponseAction.REVOKE_ACCESS
            ])
        
        return ResponsePlan(
            incident_id=incident.id,
            response_type="AUTOMATED",
            priority=incident.severity,
            estimated_duration=timedelta(minutes=30),
            required_actions=required_actions,
            responsible_team=["SecurityTeam"],
            communication_plan={
                "notify_admin": True,
                "escalate_to_ciso": incident.severity == ThreatLevel.CRITICAL
            },
            containment_steps=[
                "Block malicious IP addresses",
                "Suspend affected user accounts",
                "Collect forensic evidence"
            ],
            recovery_steps=[
                "Restore from backup if needed",
                "Update security rules",
                "Monitor for reoccurrence"
            ],
            lessons_learned_template={
                "what_happened": "",
                "root_cause": "",
                "what_worked": "",
                "what_needs_improvement": "",
                "action_items": []
            }
        )
    
    async def _execute_response_action(self, action: ResponseAction, incident: SecurityIncident) -> Dict[str, Any]:
        """Execute specific response action."""
        result = {"success": False, "details": {}}
        
        try:
            if action == ResponseAction.BLOCK_IP:
                # Block IPs from incident events
                blocked_ips = []
                for event in incident.events:
                    if event.source_ip:
                        await self._block_ip_address(event.source_ip)
                        blocked_ips.append(event.source_ip)
                
                result["success"] = True
                result["details"]["blocked_ips"] = blocked_ips
                
            elif action == ResponseAction.SUSPEND_USER:
                # Suspend affected users
                suspended_users = []
                for user_id in incident.affected_users:
                    # In production, actually suspend user accounts
                    suspended_users.append(str(user_id))
                
                result["success"] = True
                result["details"]["suspended_users"] = suspended_users
                
            elif action == ResponseAction.ALERT_ADMIN:
                # Send alert to administrators
                # In production, integrate with alerting system
                result["success"] = True
                result["details"]["alert_sent"] = True
                
            elif action == ResponseAction.COLLECT_EVIDENCE:
                # Collect forensic evidence
                evidence = {
                    "events": [e.to_dict() for e in incident.events],
                    "timestamp": datetime.utcnow().isoformat(),
                    "incident_id": str(incident.id)
                }
                
                incident.evidence.append(evidence)
                result["success"] = True
                result["details"]["evidence_collected"] = True
            
        except Exception as e:
            result["details"]["error"] = str(e)
        
        return result
    
    def _initialize_detection_rules(self) -> None:
        """Initialize threat detection rules."""
        self.detection_rules = {
            "brute_force": {
                "threshold": 10,
                "window": 300,  # 5 minutes
                "action": "block_ip"
            },
            "sql_injection": {
                "patterns": [
                    r"union\s+select",
                    r"or\s+1\s*=\s*1",
                    r"drop\s+table"
                ],
                "action": "alert_admin"
            },
            "xss_attack": {
                "patterns": [
                    r"<script>",
                    r"javascript:",
                    r"onerror="
                ],
                "action": "block_request"
            }
        }
    
    def _initialize_response_handlers(self) -> None:
        """Initialize automated response handlers."""
        self.response_handlers = {
            ThreatType.BRUTE_FORCE: self._handle_brute_force,
            ThreatType.SQL_INJECTION: self._handle_sql_injection,
            ThreatType.XSS_ATTACK: self._handle_xss_attack,
            ThreatType.DATA_EXFILTRATION: self._handle_data_exfiltration
        }
    
    async def _handle_brute_force(self, event: SecurityEvent) -> None:
        """Handle brute force attack."""
        if event.source_ip:
            await self._block_ip_address(event.source_ip)
    
    async def _handle_sql_injection(self, event: SecurityEvent) -> None:
        """Handle SQL injection attempt."""
        if event.source_ip:
            await self._block_ip_address(event.source_ip)
        # Additional handling like alerting security team
    
    async def _handle_xss_attack(self, event: SecurityEvent) -> None:
        """Handle XSS attack attempt."""
        # Block the specific request/session
        pass
    
    async def _handle_data_exfiltration(self, event: SecurityEvent) -> None:
        """Handle data exfiltration attempt."""
        if event.user_id:
            self.rate_limited_users[event.user_id] = datetime.utcnow() + timedelta(hours=24)
        if event.source_ip:
            await self._block_ip_address(event.source_ip)
    
    def _start_background_monitoring(self) -> None:
        """Start background monitoring tasks."""
        # Real-time monitoring task
        async def monitoring_task():
            while self.monitoring_active:
                try:
                    # Update baselines
                    await self._update_behavioral_baselines()
                    
                    # Refresh threat intelligence
                    await self._refresh_threat_intelligence()
                    
                    # Clean up old incidents
                    await self._cleanup_old_incidents()
                    
                    await asyncio.sleep(3600)  # Run hourly
                except Exception as e:
                    logger.error(f"Monitoring task failed: {e}")
        
        self._background_tasks = [
            asyncio.create_task(monitoring_task())
        ]
    
    async def _update_behavioral_baselines(self) -> None:
        """Update user behavioral baselines."""
        # Simplified baseline update
        # In production, use machine learning algorithms
        pass
    
    async def _refresh_threat_intelligence(self) -> None:
        """Refresh threat intelligence feeds."""
        # In production, integrate with threat intelligence feeds
        pass
    
    async def _cleanup_old_incidents(self) -> None:
        """Clean up old resolved incidents."""
        cutoff_date = datetime.utcnow() - timedelta(days=self.config["max_incident_age_days"])
        
        incidents_to_remove = []
        for incident_id, incident in self.active_incidents.items():
            if (incident.status == IncidentStatus.CLOSED and 
                incident.updated_at < cutoff_date):
                incidents_to_remove.append(incident_id)
        
        for incident_id in incidents_to_remove:
            del self.active_incidents[incident_id]
            self.metrics.active_incidents -= 1


# Factory function
async def create_security_monitoring_system(
    db_session: AsyncSession,
    security_audit: SecurityAuditSystem,
    compliance_audit: ComplianceAuditSystem,
    redis_client: Optional[Redis] = None
) -> SecurityMonitoringSystem:
    """
    Create security monitoring system instance.
    
    Args:
        db_session: Database session
        security_audit: Security audit system
        compliance_audit: Compliance audit system
        redis_client: Redis client for real-time processing
        
    Returns:
        SecurityMonitoringSystem instance
    """
    return SecurityMonitoringSystem(db_session, security_audit, compliance_audit, redis_client)