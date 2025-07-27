"""
Security Audit System for Context Engine.

Implements comprehensive security monitoring, threat detection,
and compliance validation for context access patterns.
"""

import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field
import logging
import asyncio
from collections import defaultdict, Counter

from sqlalchemy import select, and_, or_, func, desc, text
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.context import Context
from ..models.agent import Agent
from ..core.access_control import AccessControlManager, AccessLevel, Permission


logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Security threat severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AuditEventType(Enum):
    """Types of security audit events."""
    UNAUTHORIZED_ACCESS = "UNAUTHORIZED_ACCESS"
    PRIVILEGE_ESCALATION = "PRIVILEGE_ESCALATION"
    DATA_EXFILTRATION = "DATA_EXFILTRATION"
    SUSPICIOUS_PATTERN = "SUSPICIOUS_PATTERN"
    POLICY_VIOLATION = "POLICY_VIOLATION"
    BULK_ACCESS = "BULK_ACCESS"
    OFF_HOURS_ACCESS = "OFF_HOURS_ACCESS"
    RAPID_FIRE_QUERIES = "RAPID_FIRE_QUERIES"
    CROSS_AGENT_ABUSE = "CROSS_AGENT_ABUSE"
    RETENTION_VIOLATION = "RETENTION_VIOLATION"


@dataclass
class SecurityEvent:
    """Represents a security audit event."""
    id: uuid.UUID
    event_type: AuditEventType
    threat_level: ThreatLevel
    agent_id: Optional[uuid.UUID]
    context_id: Optional[uuid.UUID]
    session_id: Optional[uuid.UUID]
    description: str
    details: Dict[str, Any]
    timestamp: datetime
    resolved: bool = False
    false_positive: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "event_type": self.event_type.value,
            "threat_level": self.threat_level.value,
            "agent_id": str(self.agent_id) if self.agent_id else None,
            "context_id": str(self.context_id) if self.context_id else None,
            "session_id": str(self.session_id) if self.session_id else None,
            "description": self.description,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
            "false_positive": self.false_positive
        }


@dataclass
class AccessPattern:
    """Represents agent access patterns for analysis."""
    agent_id: uuid.UUID
    total_accesses: int = 0
    unique_contexts: Set[uuid.UUID] = field(default_factory=set)
    access_times: List[datetime] = field(default_factory=list)
    failed_attempts: int = 0
    cross_agent_accesses: int = 0
    off_hours_accesses: int = 0
    bulk_access_events: int = 0
    context_types_accessed: Set[str] = field(default_factory=set)
    
    def calculate_risk_score(self) -> float:
        """Calculate risk score based on access patterns."""
        risk_score = 0.0
        
        # High volume access
        if self.total_accesses > 1000:
            risk_score += 0.3
        elif self.total_accesses > 500:
            risk_score += 0.2
        
        # High failure rate
        if self.total_accesses > 0:
            failure_rate = self.failed_attempts / self.total_accesses
            if failure_rate > 0.3:
                risk_score += 0.4
            elif failure_rate > 0.1:
                risk_score += 0.2
        
        # Off-hours access
        if self.off_hours_accesses > 10:
            risk_score += 0.3
        elif self.off_hours_accesses > 5:
            risk_score += 0.1
        
        # Bulk access patterns
        if self.bulk_access_events > 5:
            risk_score += 0.3
        
        # Cross-agent access abuse
        if self.cross_agent_accesses > self.total_accesses * 0.8:
            risk_score += 0.4
        
        return min(1.0, risk_score)


class SecurityAuditSystem:
    """
    Comprehensive security audit system for Context Engine.
    
    Features:
    - Real-time threat detection
    - Access pattern analysis
    - Compliance monitoring
    - Automated alerting
    - Forensic investigation support
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        access_control_manager: AccessControlManager
    ):
        """
        Initialize security audit system.
        
        Args:
            db_session: Database session
            access_control_manager: Access control manager for policy checks
        """
        self.db = db_session
        self.access_control = access_control_manager
        
        # Security events storage
        self.security_events: List[SecurityEvent] = []
        self.max_events = 10000  # Keep last 10k events in memory
        
        # Pattern tracking
        self.access_patterns: Dict[uuid.UUID, AccessPattern] = {}
        self.session_tracking: Dict[uuid.UUID, Dict[str, Any]] = {}
        
        # Configuration
        self.config = {
            "bulk_access_threshold": 50,  # Contexts accessed in short time
            "rapid_fire_threshold": 10,   # Queries per minute
            "off_hours_start": 22,        # 10 PM
            "off_hours_end": 6,           # 6 AM
            "max_cross_agent_ratio": 0.7, # Max ratio of cross-agent accesses
            "retention_days": 90,         # Data retention policy
            "failed_access_threshold": 10 # Failed attempts before alert
        }
        
        # Real-time monitoring
        self.monitoring_active = True
        self.alert_callbacks: List[callable] = []
    
    async def audit_context_access(
        self,
        context_id: uuid.UUID,
        agent_id: uuid.UUID,
        session_id: Optional[uuid.UUID],
        access_granted: bool,
        permission: Permission,
        access_time: Optional[datetime] = None
    ) -> Optional[SecurityEvent]:
        """
        Audit a context access event and detect security issues.
        
        Args:
            context_id: Context being accessed
            agent_id: Agent requesting access
            session_id: Session context
            access_granted: Whether access was granted
            permission: Type of permission requested
            access_time: Time of access (defaults to now)
            
        Returns:
            SecurityEvent if threat detected, None otherwise
        """
        access_time = access_time or datetime.utcnow()
        
        try:
            # Update access patterns
            await self._update_access_patterns(
                agent_id, context_id, session_id, access_granted, access_time
            )
            
            # Check for immediate threats
            security_event = await self._analyze_access_event(
                context_id, agent_id, session_id, access_granted, permission, access_time
            )
            
            if security_event:
                await self._handle_security_event(security_event)
                return security_event
            
            # Check for pattern-based threats
            pattern_event = await self._analyze_access_patterns(agent_id)
            if pattern_event:
                await self._handle_security_event(pattern_event)
                return pattern_event
            
            return None
            
        except Exception as e:
            logger.error(f"Error during security audit: {e}")
            return None
    
    async def perform_security_scan(
        self,
        scan_type: str = "full",
        hours_back: int = 24
    ) -> Dict[str, Any]:
        """
        Perform comprehensive security scan.
        
        Args:
            scan_type: Type of scan (full, quick, targeted)
            hours_back: Hours of data to analyze
            
        Returns:
            Security scan report
        """
        logger.info(f"Starting {scan_type} security scan for last {hours_back} hours")
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        scan_results = {
            "scan_timestamp": datetime.utcnow().isoformat(),
            "scan_type": scan_type,
            "hours_analyzed": hours_back,
            "threats_detected": [],
            "vulnerabilities": [],
            "compliance_issues": [],
            "recommendations": []
        }
        
        try:
            # Analyze access patterns
            pattern_threats = await self._scan_access_patterns(cutoff_time)
            scan_results["threats_detected"].extend(pattern_threats)
            
            # Check for data retention violations
            retention_issues = await self._scan_retention_compliance()
            scan_results["compliance_issues"].extend(retention_issues)
            
            # Analyze privilege usage
            privilege_issues = await self._scan_privilege_usage(cutoff_time)
            scan_results["vulnerabilities"].extend(privilege_issues)
            
            # Check for unauthorized cross-agent access
            cross_agent_issues = await self._scan_cross_agent_access(cutoff_time)
            scan_results["threats_detected"].extend(cross_agent_issues)
            
            # Generate recommendations
            recommendations = self._generate_security_recommendations(scan_results)
            scan_results["recommendations"] = recommendations
            
            # Calculate overall risk score
            scan_results["overall_risk_score"] = self._calculate_overall_risk(scan_results)
            
            logger.info(f"Security scan completed: {len(scan_results['threats_detected'])} threats detected")
            
            return scan_results
            
        except Exception as e:
            logger.error(f"Security scan failed: {e}")
            scan_results["error"] = str(e)
            return scan_results
    
    async def get_security_dashboard(self) -> Dict[str, Any]:
        """
        Get real-time security dashboard data.
        
        Returns:
            Security dashboard metrics
        """
        try:
            # Recent events summary
            recent_events = [
                event for event in self.security_events
                if event.timestamp >= datetime.utcnow() - timedelta(hours=24)
            ]
            
            # Threat level distribution
            threat_distribution = Counter(event.threat_level.value for event in recent_events)
            
            # Top risk agents
            risk_agents = [
                {
                    "agent_id": str(agent_id),
                    "risk_score": pattern.calculate_risk_score(),
                    "total_accesses": pattern.total_accesses,
                    "failed_attempts": pattern.failed_attempts
                }
                for agent_id, pattern in self.access_patterns.items()
            ]
            risk_agents.sort(key=lambda x: x["risk_score"], reverse=True)
            
            # Active threats
            active_threats = [
                event.to_dict() for event in recent_events
                if not event.resolved and not event.false_positive
            ]
            
            # Access statistics
            total_accesses = sum(pattern.total_accesses for pattern in self.access_patterns.values())
            total_failures = sum(pattern.failed_attempts for pattern in self.access_patterns.values())
            
            dashboard = {
                "timestamp": datetime.utcnow().isoformat(),
                "summary": {
                    "total_agents_monitored": len(self.access_patterns),
                    "recent_events_24h": len(recent_events),
                    "active_threats": len(active_threats),
                    "total_accesses": total_accesses,
                    "total_failures": total_failures,
                    "failure_rate": total_failures / max(1, total_accesses)
                },
                "threat_distribution": dict(threat_distribution),
                "top_risk_agents": risk_agents[:10],
                "active_threats": active_threats,
                "monitoring_status": {
                    "active": self.monitoring_active,
                    "events_in_memory": len(self.security_events),
                    "patterns_tracked": len(self.access_patterns)
                }
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error generating security dashboard: {e}")
            return {"error": str(e)}
    
    async def investigate_security_incident(
        self,
        incident_id: Optional[uuid.UUID] = None,
        agent_id: Optional[uuid.UUID] = None,
        context_id: Optional[uuid.UUID] = None,
        time_range_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Conduct forensic investigation of security incident.
        
        Args:
            incident_id: Specific incident to investigate
            agent_id: Agent to investigate
            context_id: Context to investigate
            time_range_hours: Time range for investigation
            
        Returns:
            Investigation report
        """
        logger.info(f"Starting security investigation: incident={incident_id}, agent={agent_id}")
        
        cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)
        
        investigation = {
            "investigation_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "parameters": {
                "incident_id": str(incident_id) if incident_id else None,
                "agent_id": str(agent_id) if agent_id else None,
                "context_id": str(context_id) if context_id else None,
                "time_range_hours": time_range_hours
            },
            "timeline": [],
            "related_events": [],
            "affected_contexts": [],
            "risk_assessment": {},
            "recommended_actions": []
        }
        
        try:
            # Build timeline of events
            relevant_events = []
            
            for event in self.security_events:
                if event.timestamp < cutoff_time:
                    continue
                
                include_event = False
                
                if incident_id and event.id == incident_id:
                    include_event = True
                elif agent_id and event.agent_id == agent_id:
                    include_event = True
                elif context_id and event.context_id == context_id:
                    include_event = True
                
                if include_event:
                    relevant_events.append(event)
            
            # Sort by timestamp
            relevant_events.sort(key=lambda e: e.timestamp)
            
            # Build investigation timeline
            investigation["timeline"] = [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type.value,
                    "threat_level": event.threat_level.value,
                    "description": event.description,
                    "details": event.details
                }
                for event in relevant_events
            ]
            
            # Analyze patterns in the events
            if agent_id and agent_id in self.access_patterns:
                pattern = self.access_patterns[agent_id]
                investigation["risk_assessment"] = {
                    "risk_score": pattern.calculate_risk_score(),
                    "access_volume": pattern.total_accesses,
                    "failure_rate": pattern.failed_attempts / max(1, pattern.total_accesses),
                    "off_hours_activity": pattern.off_hours_accesses,
                    "cross_agent_activity": pattern.cross_agent_accesses
                }
            
            # Get affected contexts
            if agent_id:
                affected_contexts = await self._get_agent_accessed_contexts(agent_id, cutoff_time)
                investigation["affected_contexts"] = [
                    {"context_id": str(ctx_id), "access_count": count}
                    for ctx_id, count in affected_contexts.items()
                ]
            
            # Generate recommended actions
            investigation["recommended_actions"] = self._generate_incident_recommendations(
                relevant_events, investigation["risk_assessment"]
            )
            
            return investigation
            
        except Exception as e:
            logger.error(f"Security investigation failed: {e}")
            investigation["error"] = str(e)
            return investigation
    
    async def _update_access_patterns(
        self,
        agent_id: uuid.UUID,
        context_id: uuid.UUID,
        session_id: Optional[uuid.UUID],
        access_granted: bool,
        access_time: datetime
    ) -> None:
        """Update access patterns for pattern analysis."""
        if agent_id not in self.access_patterns:
            self.access_patterns[agent_id] = AccessPattern(agent_id=agent_id)
        
        pattern = self.access_patterns[agent_id]
        
        if access_granted:
            pattern.total_accesses += 1
            pattern.unique_contexts.add(context_id)
            pattern.access_times.append(access_time)
            
            # Check if this is off-hours access
            hour = access_time.hour
            if hour >= self.config["off_hours_start"] or hour <= self.config["off_hours_end"]:
                pattern.off_hours_accesses += 1
            
            # Check if this is cross-agent access
            context = await self.db.get(Context, context_id)
            if context and context.agent_id != agent_id:
                pattern.cross_agent_accesses += 1
            
            # Track context types
            if context and context.context_type:
                pattern.context_types_accessed.add(context.context_type.value)
        else:
            pattern.failed_attempts += 1
        
        # Keep only recent access times (last 24 hours)
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        pattern.access_times = [t for t in pattern.access_times if t >= cutoff_time]
    
    async def _analyze_access_event(
        self,
        context_id: uuid.UUID,
        agent_id: uuid.UUID,
        session_id: Optional[uuid.UUID],
        access_granted: bool,
        permission: Permission,
        access_time: datetime
    ) -> Optional[SecurityEvent]:
        """Analyze individual access event for immediate threats."""
        
        # Check for unauthorized access attempts
        if not access_granted:
            pattern = self.access_patterns.get(agent_id)
            if pattern and pattern.failed_attempts >= self.config["failed_access_threshold"]:
                return SecurityEvent(
                    id=uuid.uuid4(),
                    event_type=AuditEventType.UNAUTHORIZED_ACCESS,
                    threat_level=ThreatLevel.HIGH,
                    agent_id=agent_id,
                    context_id=context_id,
                    session_id=session_id,
                    description=f"Agent {agent_id} exceeded failed access threshold",
                    details={
                        "failed_attempts": pattern.failed_attempts,
                        "threshold": self.config["failed_access_threshold"],
                        "permission": permission.value
                    },
                    timestamp=access_time
                )
        
        # Check for privilege escalation attempts
        if access_granted and permission in [Permission.WRITE, Permission.DELETE]:
            context = await self.db.get(Context, context_id)
            if context and context.agent_id != agent_id:
                # Cross-agent write/delete attempt
                return SecurityEvent(
                    id=uuid.uuid4(),
                    event_type=AuditEventType.PRIVILEGE_ESCALATION,
                    threat_level=ThreatLevel.CRITICAL,
                    agent_id=agent_id,
                    context_id=context_id,
                    session_id=session_id,
                    description=f"Cross-agent {permission.value} attempt on context {context_id}",
                    details={
                        "context_owner": str(context.agent_id),
                        "requesting_agent": str(agent_id),
                        "permission": permission.value
                    },
                    timestamp=access_time
                )
        
        return None
    
    async def _analyze_access_patterns(self, agent_id: uuid.UUID) -> Optional[SecurityEvent]:
        """Analyze access patterns for suspicious behavior."""
        if agent_id not in self.access_patterns:
            return None
        
        pattern = self.access_patterns[agent_id]
        
        # Check for rapid-fire queries
        recent_accesses = [
            t for t in pattern.access_times
            if t >= datetime.utcnow() - timedelta(minutes=1)
        ]
        
        if len(recent_accesses) >= self.config["rapid_fire_threshold"]:
            return SecurityEvent(
                id=uuid.uuid4(),
                event_type=AuditEventType.RAPID_FIRE_QUERIES,
                threat_level=ThreatLevel.MEDIUM,
                agent_id=agent_id,
                context_id=None,
                session_id=None,
                description=f"Agent {agent_id} making rapid-fire queries",
                details={
                    "queries_per_minute": len(recent_accesses),
                    "threshold": self.config["rapid_fire_threshold"]
                },
                timestamp=datetime.utcnow()
            )
        
        # Check for bulk access patterns
        recent_contexts = len([
            t for t in pattern.access_times
            if t >= datetime.utcnow() - timedelta(minutes=10)
        ])
        
        if recent_contexts >= self.config["bulk_access_threshold"]:
            return SecurityEvent(
                id=uuid.uuid4(),
                event_type=AuditEventType.BULK_ACCESS,
                threat_level=ThreatLevel.HIGH,
                agent_id=agent_id,
                context_id=None,
                session_id=None,
                description=f"Agent {agent_id} bulk accessing contexts",
                details={
                    "contexts_accessed": recent_contexts,
                    "time_window_minutes": 10,
                    "threshold": self.config["bulk_access_threshold"]
                },
                timestamp=datetime.utcnow()
            )
        
        return None
    
    async def _handle_security_event(self, event: SecurityEvent) -> None:
        """Handle a detected security event."""
        self.security_events.append(event)
        
        # Manage memory usage
        if len(self.security_events) > self.max_events:
            self.security_events = self.security_events[-self.max_events:]
        
        # Log the event
        logger.warning(
            f"Security event detected: {event.event_type.value} "
            f"(Threat Level: {event.threat_level.value}) - {event.description}"
        )
        
        # Trigger alerts for high/critical threats
        if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            await self._trigger_security_alert(event)
    
    async def _trigger_security_alert(self, event: SecurityEvent) -> None:
        """Trigger security alert for high-priority events."""
        alert_data = {
            "event": event.to_dict(),
            "alert_level": event.threat_level.value,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Call registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_data)
                else:
                    callback(alert_data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    async def _scan_access_patterns(self, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Scan for suspicious access patterns."""
        threats = []
        
        for agent_id, pattern in self.access_patterns.items():
            risk_score = pattern.calculate_risk_score()
            
            if risk_score > 0.7:  # High risk threshold
                threats.append({
                    "type": "suspicious_access_pattern",
                    "agent_id": str(agent_id),
                    "risk_score": risk_score,
                    "details": {
                        "total_accesses": pattern.total_accesses,
                        "failed_attempts": pattern.failed_attempts,
                        "off_hours_accesses": pattern.off_hours_accesses,
                        "cross_agent_accesses": pattern.cross_agent_accesses
                    }
                })
        
        return threats
    
    async def _scan_retention_compliance(self) -> List[Dict[str, Any]]:
        """Scan for data retention compliance issues."""
        issues = []
        
        try:
            # Check for old contexts that should be archived
            retention_cutoff = datetime.utcnow() - timedelta(days=self.config["retention_days"])
            
            old_contexts_query = select(func.count(Context.id)).where(
                and_(
                    Context.created_at < retention_cutoff,
                    or_(
                        Context.context_metadata.op('->>')('archived') != 'true',
                        Context.context_metadata.op('->>')('archived').is_(None)
                    )
                )
            )
            
            result = await self.db.execute(old_contexts_query)
            old_context_count = result.scalar() or 0
            
            if old_context_count > 0:
                issues.append({
                    "type": "retention_violation",
                    "description": f"{old_context_count} contexts exceed retention policy",
                    "count": old_context_count,
                    "retention_days": self.config["retention_days"]
                })
        
        except Exception as e:
            logger.error(f"Retention compliance scan failed: {e}")
        
        return issues
    
    async def _scan_privilege_usage(self, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Scan for privilege usage anomalies."""
        vulnerabilities = []
        
        # This would typically analyze privilege escalation patterns
        # For now, we'll check for excessive cross-agent access
        
        for agent_id, pattern in self.access_patterns.items():
            if pattern.total_accesses > 0:
                cross_agent_ratio = pattern.cross_agent_accesses / pattern.total_accesses
                
                if cross_agent_ratio > self.config["max_cross_agent_ratio"]:
                    vulnerabilities.append({
                        "type": "excessive_cross_agent_access",
                        "agent_id": str(agent_id),
                        "cross_agent_ratio": cross_agent_ratio,
                        "threshold": self.config["max_cross_agent_ratio"]
                    })
        
        return vulnerabilities
    
    async def _scan_cross_agent_access(self, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Scan for unauthorized cross-agent access."""
        threats = []
        
        # Analyze cross-agent access patterns for abuse
        for agent_id, pattern in self.access_patterns.items():
            if pattern.cross_agent_accesses > 100:  # High volume threshold
                threats.append({
                    "type": "potential_data_exfiltration",
                    "agent_id": str(agent_id),
                    "cross_agent_accesses": pattern.cross_agent_accesses,
                    "unique_contexts": len(pattern.unique_contexts)
                })
        
        return threats
    
    async def _get_agent_accessed_contexts(
        self, 
        agent_id: uuid.UUID, 
        cutoff_time: datetime
    ) -> Dict[uuid.UUID, int]:
        """Get contexts accessed by agent in time window."""
        if agent_id not in self.access_patterns:
            return {}
        
        # This is simplified - in production you'd query the analytics database
        pattern = self.access_patterns[agent_id]
        return {ctx_id: 1 for ctx_id in pattern.unique_contexts}  # Simplified count
    
    def _generate_security_recommendations(self, scan_results: Dict[str, Any]) -> List[str]:
        """Generate security recommendations based on scan results."""
        recommendations = []
        
        threat_count = len(scan_results["threats_detected"])
        vulnerability_count = len(scan_results["vulnerabilities"])
        compliance_count = len(scan_results["compliance_issues"])
        
        if threat_count > 0:
            recommendations.append(f"Investigate {threat_count} detected security threats immediately")
        
        if vulnerability_count > 0:
            recommendations.append(f"Address {vulnerability_count} security vulnerabilities")
        
        if compliance_count > 0:
            recommendations.append(f"Resolve {compliance_count} compliance issues")
        
        # Specific recommendations based on threat types
        threat_types = set()
        for threat in scan_results["threats_detected"]:
            threat_types.add(threat.get("type", "unknown"))
        
        if "suspicious_access_pattern" in threat_types:
            recommendations.append("Review agent access patterns and implement additional monitoring")
        
        if "potential_data_exfiltration" in threat_types:
            recommendations.append("Implement stricter cross-agent access controls")
        
        if not recommendations:
            recommendations.append("No immediate security actions required - continue monitoring")
        
        return recommendations
    
    def _generate_incident_recommendations(
        self, 
        events: List[SecurityEvent], 
        risk_assessment: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for incident response."""
        recommendations = []
        
        if not events:
            return ["No security events found in investigation timeframe"]
        
        # Analyze event severity
        critical_events = [e for e in events if e.threat_level == ThreatLevel.CRITICAL]
        high_events = [e for e in events if e.threat_level == ThreatLevel.HIGH]
        
        if critical_events:
            recommendations.append("Immediate action required - critical security events detected")
            recommendations.append("Consider temporarily suspending agent access pending investigation")
        
        if high_events:
            recommendations.append("High-priority security events require investigation")
        
        # Risk-based recommendations
        risk_score = risk_assessment.get("risk_score", 0)
        if risk_score > 0.8:
            recommendations.append("Agent poses high security risk - implement enhanced monitoring")
        elif risk_score > 0.5:
            recommendations.append("Agent poses moderate security risk - review access patterns")
        
        return recommendations
    
    def _calculate_overall_risk(self, scan_results: Dict[str, Any]) -> float:
        """Calculate overall security risk score."""
        risk_score = 0.0
        
        # Weight threats by severity
        for threat in scan_results["threats_detected"]:
            if threat.get("type") == "potential_data_exfiltration":
                risk_score += 0.4
            elif threat.get("type") == "suspicious_access_pattern":
                risk_score += 0.3
            else:
                risk_score += 0.2
        
        # Add vulnerability weight
        risk_score += len(scan_results["vulnerabilities"]) * 0.1
        
        # Add compliance weight
        risk_score += len(scan_results["compliance_issues"]) * 0.05
        
        return min(1.0, risk_score)
    
    def register_alert_callback(self, callback: callable) -> None:
        """Register callback for security alerts."""
        self.alert_callbacks.append(callback)
    
    def set_monitoring_active(self, active: bool) -> None:
        """Enable or disable real-time monitoring."""
        self.monitoring_active = active
        logger.info(f"Security monitoring {'enabled' if active else 'disabled'}")


# Factory function
async def create_security_audit_system(
    db_session: AsyncSession,
    access_control_manager: AccessControlManager
) -> SecurityAuditSystem:
    """
    Create security audit system instance.
    
    Args:
        db_session: Database session
        access_control_manager: Access control manager
        
    Returns:
        SecurityAuditSystem instance
    """
    return SecurityAuditSystem(db_session, access_control_manager)