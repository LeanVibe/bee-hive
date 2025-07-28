"""
Enhanced Security Audit System for LeanVibe Agent Hive 2.0.

This system extends the existing SecurityAuditSystem with advanced forensic capabilities,
comprehensive logging, threat correlation, and integration with the advanced security components.

Features:
- Advanced forensic analysis and investigation workflows
- Comprehensive security event correlation and timeline analysis
- Real-time threat intelligence integration
- Advanced query and search capabilities for security events
- Automated incident response workflows
- Integration with AdvancedSecurityValidator and ThreatDetectionEngine
- Comprehensive compliance reporting and audit trails
"""

import asyncio
import uuid
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque, Counter
import logging

import structlog
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc, text

from .security_audit import (
    SecurityAuditSystem, ThreatLevel, AuditEventType, SecurityEvent, AccessPattern
)
from .advanced_security_validator import (
    CommandContext, SecurityAnalysisResult, ThreatCategory, ThreatSignature
)
from .threat_detection_engine import ThreatDetection, ThreatType, DetectionMethod
from .security_policy_engine import SecurityPolicy, PolicyEvaluationResult
from .access_control import AccessControlManager
from ..models.context import Context
from ..models.agent import Agent

logger = structlog.get_logger()


class ForensicEventType(Enum):
    """Types of forensic events for investigation."""
    COMMAND_EXECUTION = "COMMAND_EXECUTION"
    SECURITY_VIOLATION = "SECURITY_VIOLATION"
    POLICY_TRIGGER = "POLICY_TRIGGER"
    THREAT_DETECTION = "THREAT_DETECTION"
    BEHAVIORAL_ANOMALY = "BEHAVIORAL_ANOMALY"
    ACCESS_ATTEMPT = "ACCESS_ATTEMPT"
    DATA_ACCESS = "DATA_ACCESS"
    SYSTEM_EVENT = "SYSTEM_EVENT"
    AUTHENTICATION_EVENT = "AUTHENTICATION_EVENT"
    AUTHORIZATION_EVENT = "AUTHORIZATION_EVENT"


class InvestigationStatus(Enum):
    """Status of security investigations."""
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    ESCALATED = "ESCALATED"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"
    FALSE_POSITIVE = "FALSE_POSITIVE"


@dataclass
class ForensicEvent:
    """Enhanced forensic event with comprehensive metadata."""
    id: str
    event_type: ForensicEventType
    severity: ThreatLevel
    agent_id: Optional[uuid.UUID]
    session_id: Optional[uuid.UUID]
    timestamp: datetime
    
    # Event details
    description: str
    raw_data: Dict[str, Any]
    processed_data: Dict[str, Any]
    
    # Context and attribution
    command_hash: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    geo_location: Optional[Dict[str, str]] = None
    
    # Analysis results
    threat_indicators: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    confidence: float = 0.0
    
    # Correlation and relationships
    related_events: List[str] = field(default_factory=list)
    parent_investigation: Optional[str] = None
    correlation_score: float = 0.0
    
    # Investigation metadata
    investigated: bool = False
    investigation_notes: str = ""
    false_positive: bool = False
    
    # Compliance and tags
    compliance_tags: List[str] = field(default_factory=list)
    event_tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "agent_id": str(self.agent_id) if self.agent_id else None,
            "session_id": str(self.session_id) if self.session_id else None,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            "raw_data": self.raw_data,
            "processed_data": self.processed_data,
            "command_hash": self.command_hash,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "geo_location": self.geo_location,
            "threat_indicators": self.threat_indicators,
            "risk_score": self.risk_score,
            "confidence": self.confidence,
            "related_events": self.related_events,
            "parent_investigation": self.parent_investigation,
            "correlation_score": self.correlation_score,
            "investigated": self.investigated,
            "investigation_notes": self.investigation_notes,
            "false_positive": self.false_positive,
            "compliance_tags": self.compliance_tags,
            "event_tags": self.event_tags
        }


@dataclass
class SecurityInvestigation:
    """Security investigation with forensic analysis."""
    id: str
    title: str
    description: str
    status: InvestigationStatus
    priority: ThreatLevel
    
    # Investigation scope
    affected_agents: List[uuid.UUID]
    time_range_start: datetime
    time_range_end: datetime
    
    # Events and evidence
    related_events: List[str] = field(default_factory=list)
    evidence_collected: List[Dict[str, Any]] = field(default_factory=list)
    
    # Analysis and findings
    threat_assessment: Dict[str, Any] = field(default_factory=dict)
    attack_timeline: List[Dict[str, Any]] = field(default_factory=list)
    indicators_of_compromise: List[str] = field(default_factory=list)
    
    # Investigation metadata
    created_by: str = "system"
    assigned_to: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Resolution
    resolution_summary: Optional[str] = None
    remediation_actions: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "affected_agents": [str(agent_id) for agent_id in self.affected_agents],
            "time_range_start": self.time_range_start.isoformat(),
            "time_range_end": self.time_range_end.isoformat(),
            "related_events": self.related_events,
            "evidence_collected": self.evidence_collected,
            "threat_assessment": self.threat_assessment,
            "attack_timeline": self.attack_timeline,
            "indicators_of_compromise": self.indicators_of_compromise,
            "created_by": self.created_by,
            "assigned_to": self.assigned_to,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "resolution_summary": self.resolution_summary,
            "remediation_actions": self.remediation_actions,
            "lessons_learned": self.lessons_learned
        }


class ForensicAnalyzer:
    """Advanced forensic analysis engine."""
    
    def __init__(self):
        self.analysis_techniques = {
            "timeline_analysis": self._perform_timeline_analysis,
            "correlation_analysis": self._perform_correlation_analysis,
            "behavioral_analysis": self._perform_behavioral_analysis,
            "pattern_analysis": self._perform_pattern_analysis,
            "attribution_analysis": self._perform_attribution_analysis
        }
    
    async def analyze_events(
        self,
        events: List[ForensicEvent],
        analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Perform forensic analysis on a collection of events.
        
        Args:
            events: List of forensic events to analyze
            analysis_type: Type of analysis to perform
            
        Returns:
            Forensic analysis results
        """
        if not events:
            return {"error": "No events provided for analysis"}
        
        analysis_results = {
            "analysis_type": analysis_type,
            "event_count": len(events),
            "time_range": {
                "start": min(e.timestamp for e in events).isoformat(),
                "end": max(e.timestamp for e in events).isoformat()
            },
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "findings": {}
        }
        
        try:
            if analysis_type == "comprehensive":
                # Run all analysis techniques
                for technique_name, technique_func in self.analysis_techniques.items():
                    try:
                        result = await technique_func(events)
                        analysis_results["findings"][technique_name] = result
                    except Exception as e:
                        logger.error(f"Forensic analysis error for {technique_name}: {e}")
                        analysis_results["findings"][technique_name] = {"error": str(e)}
            else:
                # Run specific analysis technique
                if analysis_type in self.analysis_techniques:
                    result = await self.analysis_techniques[analysis_type](events)
                    analysis_results["findings"][analysis_type] = result
                else:
                    analysis_results["error"] = f"Unknown analysis type: {analysis_type}"
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Forensic analysis failed: {e}")
            analysis_results["error"] = str(e)
            return analysis_results
    
    async def _perform_timeline_analysis(self, events: List[ForensicEvent]) -> Dict[str, Any]:
        """Perform timeline analysis on events."""
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        # Create timeline
        timeline = []
        for event in sorted_events:
            timeline.append({
                "timestamp": event.timestamp.isoformat(),
                "event_id": event.id,
                "event_type": event.event_type.value,
                "severity": event.severity.value,
                "description": event.description,
                "agent_id": str(event.agent_id) if event.agent_id else None,
                "risk_score": event.risk_score
            })
        
        # Identify time gaps and clusters
        time_gaps = []
        event_clusters = []
        
        if len(sorted_events) > 1:
            for i in range(1, len(sorted_events)):
                gap = sorted_events[i].timestamp - sorted_events[i-1].timestamp
                if gap.total_seconds() > 3600:  # 1 hour gap
                    time_gaps.append({
                        "start": sorted_events[i-1].timestamp.isoformat(),
                        "end": sorted_events[i].timestamp.isoformat(),
                        "duration_hours": gap.total_seconds() / 3600
                    })
        
        # Find event clusters (events within 5 minutes)
        current_cluster = []
        cluster_threshold = timedelta(minutes=5)
        
        for event in sorted_events:
            if not current_cluster:
                current_cluster.append(event)
            else:
                time_diff = event.timestamp - current_cluster[-1].timestamp
                if time_diff <= cluster_threshold:
                    current_cluster.append(event)
                else:
                    if len(current_cluster) > 1:
                        event_clusters.append({
                            "start_time": current_cluster[0].timestamp.isoformat(),
                            "end_time": current_cluster[-1].timestamp.isoformat(),
                            "event_count": len(current_cluster),
                            "event_types": list(set(e.event_type.value for e in current_cluster))
                        })
                    current_cluster = [event]
        
        # Add final cluster if exists
        if len(current_cluster) > 1:
            event_clusters.append({
                "start_time": current_cluster[0].timestamp.isoformat(),
                "end_time": current_cluster[-1].timestamp.isoformat(),
                "event_count": len(current_cluster),
                "event_types": list(set(e.event_type.value for e in current_cluster))
            })
        
        return {
            "timeline": timeline,
            "time_gaps": time_gaps,
            "event_clusters": event_clusters,
            "total_duration_hours": (
                sorted_events[-1].timestamp - sorted_events[0].timestamp
            ).total_seconds() / 3600 if len(sorted_events) > 1 else 0
        }
    
    async def _perform_correlation_analysis(self, events: List[ForensicEvent]) -> Dict[str, Any]:
        """Perform correlation analysis on events."""
        
        correlations = []
        
        # Group events by various attributes
        agent_groups = defaultdict(list)
        session_groups = defaultdict(list)
        ip_groups = defaultdict(list)
        
        for event in events:
            if event.agent_id:
                agent_groups[event.agent_id].append(event)
            if event.session_id:
                session_groups[event.session_id].append(event)
            if event.ip_address:
                ip_groups[event.ip_address].append(event)
        
        # Analyze agent-based correlations
        for agent_id, agent_events in agent_groups.items():
            if len(agent_events) > 1:
                severity_counts = Counter(e.severity.value for e in agent_events)
                event_type_counts = Counter(e.event_type.value for e in agent_events)
                
                correlations.append({
                    "correlation_type": "agent_activity",
                    "agent_id": str(agent_id),
                    "event_count": len(agent_events),
                    "severity_distribution": dict(severity_counts),
                    "event_type_distribution": dict(event_type_counts),
                    "risk_score": np.mean([e.risk_score for e in agent_events]),
                    "time_span_hours": (
                        max(e.timestamp for e in agent_events) - 
                        min(e.timestamp for e in agent_events)
                    ).total_seconds() / 3600
                })
        
        # Analyze session-based correlations
        for session_id, session_events in session_groups.items():
            if len(session_events) > 1:
                correlations.append({
                    "correlation_type": "session_activity",
                    "session_id": str(session_id),
                    "event_count": len(session_events),
                    "unique_agents": len(set(e.agent_id for e in session_events if e.agent_id)),
                    "severity_distribution": dict(Counter(e.severity.value for e in session_events)),
                    "avg_risk_score": np.mean([e.risk_score for e in session_events])
                })
        
        # Analyze IP-based correlations
        for ip_address, ip_events in ip_groups.items():
            if len(ip_events) > 1:
                correlations.append({
                    "correlation_type": "ip_activity",
                    "ip_address": ip_address,
                    "event_count": len(ip_events),
                    "unique_agents": len(set(e.agent_id for e in ip_events if e.agent_id)),
                    "unique_sessions": len(set(e.session_id for e in ip_events if e.session_id)),
                    "avg_risk_score": np.mean([e.risk_score for e in ip_events])
                })
        
        return {
            "correlations": correlations,
            "correlation_count": len(correlations),
            "high_risk_correlations": [
                c for c in correlations 
                if c.get("avg_risk_score", 0) > 0.7 or c.get("risk_score", 0) > 0.7
            ]
        }
    
    async def _perform_behavioral_analysis(self, events: List[ForensicEvent]) -> Dict[str, Any]:
        """Perform behavioral analysis on events."""
        
        # Analyze event patterns by hour
        hourly_distribution = defaultdict(int)
        daily_distribution = defaultdict(int)
        
        for event in events:
            hourly_distribution[event.timestamp.hour] += 1
            daily_distribution[event.timestamp.date().isoformat()] += 1
        
        # Identify anomalous time periods
        if len(hourly_distribution) > 0:
            avg_hourly_events = np.mean(list(hourly_distribution.values()))
            anomalous_hours = [
                hour for hour, count in hourly_distribution.items()
                if count > avg_hourly_events * 2  # More than 2x average
            ]
        else:
            anomalous_hours = []
        
        # Analyze event type patterns
        event_type_patterns = Counter(e.event_type.value for e in events)
        
        # Analyze severity trends
        severity_counts = Counter(e.severity.value for e in events)
        
        # Calculate behavioral metrics
        total_events = len(events)
        high_severity_rate = (
            severity_counts.get(ThreatLevel.HIGH.value, 0) + 
            severity_counts.get(ThreatLevel.CRITICAL.value, 0)
        ) / max(total_events, 1)
        
        avg_risk_score = np.mean([e.risk_score for e in events]) if events else 0.0
        
        return {
            "hourly_distribution": dict(hourly_distribution),
            "daily_distribution": dict(daily_distribution),
            "anomalous_hours": anomalous_hours,
            "event_type_patterns": dict(event_type_patterns),
            "severity_distribution": dict(severity_counts),
            "behavioral_metrics": {
                "total_events": total_events,
                "high_severity_rate": high_severity_rate,
                "avg_risk_score": avg_risk_score,
                "unique_agents": len(set(e.agent_id for e in events if e.agent_id)),
                "unique_sessions": len(set(e.session_id for e in events if e.session_id)),
                "unique_ips": len(set(e.ip_address for e in events if e.ip_address))
            }
        }
    
    async def _perform_pattern_analysis(self, events: List[ForensicEvent]) -> Dict[str, Any]:
        """Perform pattern analysis on events."""
        
        patterns = []
        
        # Analyze command patterns (if available)
        command_hashes = []
        for event in events:
            if event.command_hash:
                command_hashes.append(event.command_hash)
        
        if command_hashes:
            command_frequency = Counter(command_hashes)
            repeated_commands = {
                hash_val: count for hash_val, count in command_frequency.items() 
                if count > 1
            }
            
            if repeated_commands:
                patterns.append({
                    "pattern_type": "repeated_commands",
                    "description": "Commands executed multiple times",
                    "pattern_data": repeated_commands,
                    "risk_indicator": len(repeated_commands) > 5
                })
        
        # Analyze threat indicator patterns
        all_indicators = []
        for event in events:
            all_indicators.extend(event.threat_indicators)
        
        if all_indicators:
            indicator_frequency = Counter(all_indicators)
            common_indicators = {
                indicator: count for indicator, count in indicator_frequency.items()
                if count > 1
            }
            
            if common_indicators:
                patterns.append({
                    "pattern_type": "threat_indicators",
                    "description": "Common threat indicators across events",
                    "pattern_data": common_indicators,
                    "risk_indicator": len(common_indicators) > 3
                })
        
        # Analyze escalation patterns
        severity_sequence = [e.severity.value for e in sorted(events, key=lambda x: x.timestamp)]
        
        escalations = 0
        for i in range(1, len(severity_sequence)):
            current_severity = ThreatLevel(severity_sequence[i])
            previous_severity = ThreatLevel(severity_sequence[i-1])
            
            # Check if severity increased
            severity_values = {ThreatLevel.LOW: 1, ThreatLevel.MEDIUM: 2, ThreatLevel.HIGH: 3, ThreatLevel.CRITICAL: 4}
            if severity_values.get(current_severity, 0) > severity_values.get(previous_severity, 0):
                escalations += 1
        
        if escalations > 0:
            patterns.append({
                "pattern_type": "severity_escalation",
                "description": "Increasing severity over time",
                "pattern_data": {"escalation_count": escalations},
                "risk_indicator": escalations > 2
            })
        
        return {
            "patterns": patterns,
            "pattern_count": len(patterns),
            "high_risk_patterns": [p for p in patterns if p.get("risk_indicator", False)]
        }
    
    async def _perform_attribution_analysis(self, events: List[ForensicEvent]) -> Dict[str, Any]:
        """Perform attribution analysis on events."""
        
        attribution_data = {
            "agent_attribution": {},
            "ip_attribution": {},
            "session_attribution": {},
            "geographic_attribution": {}
        }
        
        # Analyze agent attribution
        agent_events = defaultdict(list)
        for event in events:
            if event.agent_id:
                agent_events[event.agent_id].append(event)
        
        for agent_id, agent_event_list in agent_events.items():
            attribution_data["agent_attribution"][str(agent_id)] = {
                "event_count": len(agent_event_list),
                "severity_profile": dict(Counter(e.severity.value for e in agent_event_list)),
                "avg_risk_score": np.mean([e.risk_score for e in agent_event_list]),
                "time_span_hours": (
                    max(e.timestamp for e in agent_event_list) - 
                    min(e.timestamp for e in agent_event_list)
                ).total_seconds() / 3600 if len(agent_event_list) > 1 else 0,
                "unique_ips": len(set(e.ip_address for e in agent_event_list if e.ip_address)),
                "unique_sessions": len(set(e.session_id for e in agent_event_list if e.session_id))
            }
        
        # Analyze IP attribution
        ip_events = defaultdict(list)
        for event in events:
            if event.ip_address:
                ip_events[event.ip_address].append(event)
        
        for ip_address, ip_event_list in ip_events.items():
            attribution_data["ip_attribution"][ip_address] = {
                "event_count": len(ip_event_list),
                "unique_agents": len(set(e.agent_id for e in ip_event_list if e.agent_id)),
                "avg_risk_score": np.mean([e.risk_score for e in ip_event_list]),
                "severity_profile": dict(Counter(e.severity.value for e in ip_event_list))
            }
        
        # Geographic attribution
        geo_events = defaultdict(list)
        for event in events:
            if event.geo_location:
                country = event.geo_location.get("country", "unknown")
                geo_events[country].append(event)
        
        for country, geo_event_list in geo_events.items():
            attribution_data["geographic_attribution"][country] = {
                "event_count": len(geo_event_list),
                "unique_agents": len(set(e.agent_id for e in geo_event_list if e.agent_id)),
                "avg_risk_score": np.mean([e.risk_score for e in geo_event_list])
            }
        
        return attribution_data


class EnhancedSecurityAudit:
    """
    Enhanced Security Audit System with advanced forensic capabilities.
    
    Extends the base SecurityAuditSystem with comprehensive logging,
    forensic analysis, and investigation management capabilities.
    """
    
    def __init__(
        self,
        base_audit_system: SecurityAuditSystem,
        db_session: AsyncSession,
        access_control_manager: AccessControlManager
    ):
        self.base_audit_system = base_audit_system
        self.db = db_session
        self.access_control = access_control_manager
        
        # Forensic components
        self.forensic_analyzer = ForensicAnalyzer()
        
        # Event storage
        self.forensic_events: deque = deque(maxlen=50000)  # Keep last 50k events
        self.investigations: Dict[str, SecurityInvestigation] = {}
        
        # Event correlation
        self.correlation_cache: Dict[str, List[ForensicEvent]] = defaultdict(list)
        self.correlation_window = timedelta(hours=6)
        
        # Performance metrics
        self.metrics = {
            "forensic_events_processed": 0,
            "investigations_created": 0,
            "correlations_found": 0,
            "avg_forensic_processing_time_ms": 0.0,
            "false_positive_rate": 0.0,
            "investigation_resolution_rate": 0.0
        }
        
        # Configuration
        self.config = {
            "enable_forensic_analysis": True,
            "enable_auto_investigation": True,
            "enable_real_time_correlation": True,
            "correlation_threshold": 0.7,
            "auto_investigation_threshold": 0.8,
            "max_investigation_duration_days": 30,
            "retention_days": 90
        }
        
        # Background tasks
        self._correlation_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize the enhanced audit system."""
        # Start background tasks
        self._correlation_task = asyncio.create_task(self._correlation_worker())
        self._cleanup_task = asyncio.create_task(self._cleanup_worker())
        
        logger.info("Enhanced Security Audit System initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the enhanced audit system."""
        if self._correlation_task:
            self._correlation_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        logger.info("Enhanced Security Audit System shutdown completed")
    
    async def log_security_analysis_result(
        self,
        context: CommandContext,
        result: SecurityAnalysisResult,
        threat_detections: List[ThreatDetection] = None,
        policy_result: Optional[PolicyEvaluationResult] = None
    ) -> ForensicEvent:
        """
        Log comprehensive security analysis result as forensic event.
        
        Args:
            context: Command context
            result: Security analysis result
            threat_detections: Detected threats
            policy_result: Policy evaluation result
            
        Returns:
            Created ForensicEvent
        """
        start_time = time.time()
        threat_detections = threat_detections or []
        
        try:
            # Create forensic event
            event_id = str(uuid.uuid4())
            command_hash = hashlib.sha256(context.command.encode()).hexdigest()[:16]
            
            # Determine severity
            if result.risk_level.value == "CRITICAL":
                severity = ThreatLevel.CRITICAL
            elif result.risk_level.value == "HIGH":
                severity = ThreatLevel.HIGH
            elif result.risk_level.value == "MODERATE":
                severity = ThreatLevel.MEDIUM
            else:
                severity = ThreatLevel.LOW
            
            # Collect threat indicators
            threat_indicators = []
            threat_indicators.extend(result.risk_factors)
            threat_indicators.extend(result.behavioral_anomalies)
            threat_indicators.extend(result.matched_signatures)
            
            # Add threat detection indicators
            for detection in threat_detections:
                threat_indicators.extend(detection.indicators)
            
            # Build forensic event
            forensic_event = ForensicEvent(
                id=event_id,
                event_type=ForensicEventType.COMMAND_EXECUTION,
                severity=severity,
                agent_id=context.agent_id,
                session_id=context.session_id,
                timestamp=context.timestamp,
                description=f"Command execution analysis: {result.command_intent or 'unknown intent'}",
                raw_data={
                    "command": context.command,
                    "working_directory": context.working_directory,
                    "environment_vars": context.environment_vars,
                    "previous_commands": context.previous_commands[-5:] if context.previous_commands else []
                },
                processed_data={
                    "security_analysis": {
                        "is_safe": result.is_safe,
                        "risk_level": result.risk_level.value,
                        "threat_categories": [cat.value for cat in result.threat_categories],
                        "confidence_score": result.confidence_score,
                        "control_decision": result.control_decision.value,
                        "analysis_time_ms": result.analysis_time_ms,
                        "analysis_mode": result.analysis_mode.value
                    },
                    "threat_detections": [detection.to_dict() for detection in threat_detections],
                    "policy_evaluation": policy_result.to_dict() if policy_result else None
                },
                command_hash=command_hash,
                ip_address=context.source_ip,
                user_agent=context.user_agent,
                geo_location=context.geo_location,
                threat_indicators=threat_indicators,
                risk_score=result.confidence_score if not result.is_safe else 0.0,
                confidence=result.confidence_score,
                compliance_tags=["command_execution", "security_analysis"],
                event_tags=self._generate_event_tags(result, threat_detections)
            )
            
            # Store forensic event
            self.forensic_events.append(forensic_event)
            
            # Add to correlation cache
            if context.agent_id:
                self.correlation_cache[str(context.agent_id)].append(forensic_event)
            
            # Trigger investigation if necessary
            if self.config["enable_auto_investigation"] and result.confidence_score > self.config["auto_investigation_threshold"]:
                await self._trigger_auto_investigation(forensic_event, threat_detections)
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self._update_metrics(processing_time)
            
            # Log to base audit system
            await self.base_audit_system.audit_context_access(
                context_id=uuid.uuid4(),  # Placeholder
                agent_id=context.agent_id,
                session_id=context.session_id,
                access_granted=result.is_safe,
                permission="command_execution",
                access_time=context.timestamp
            )
            
            return forensic_event
            
        except Exception as e:
            logger.error(f"Failed to log security analysis result: {e}")
            # Return minimal event on error
            return ForensicEvent(
                id=str(uuid.uuid4()),
                event_type=ForensicEventType.SYSTEM_EVENT,
                severity=ThreatLevel.HIGH,
                agent_id=context.agent_id,
                session_id=context.session_id,
                timestamp=context.timestamp,
                description=f"Failed to log security analysis: {str(e)}",
                raw_data={"error": str(e)},
                processed_data={}
            )
    
    async def create_investigation(
        self,
        title: str,
        description: str,
        priority: ThreatLevel,
        affected_agents: List[uuid.UUID],
        time_range_start: datetime,
        time_range_end: datetime,
        created_by: str = "system"
    ) -> SecurityInvestigation:
        """
        Create a new security investigation.
        
        Args:
            title: Investigation title
            description: Investigation description
            priority: Investigation priority
            affected_agents: List of affected agent IDs
            time_range_start: Investigation time range start
            time_range_end: Investigation time range end
            created_by: Who created the investigation
            
        Returns:
            Created SecurityInvestigation
        """
        investigation_id = str(uuid.uuid4())
        
        investigation = SecurityInvestigation(
            id=investigation_id,
            title=title,
            description=description,
            status=InvestigationStatus.OPEN,
            priority=priority,
            affected_agents=affected_agents,
            time_range_start=time_range_start,
            time_range_end=time_range_end,
            created_by=created_by
        )
        
        # Collect relevant events
        relevant_events = []
        for event in self.forensic_events:
            if (event.agent_id in affected_agents and
                time_range_start <= event.timestamp <= time_range_end):
                relevant_events.append(event.id)
                event.parent_investigation = investigation_id
        
        investigation.related_events = relevant_events
        
        # Perform initial analysis
        if relevant_events:
            forensic_events = [e for e in self.forensic_events if e.id in relevant_events]
            analysis_result = await self.forensic_analyzer.analyze_events(forensic_events)
            investigation.threat_assessment = analysis_result
            
            # Extract indicators of compromise
            investigation.indicators_of_compromise = self._extract_iocs(forensic_events)
            
            # Build attack timeline
            investigation.attack_timeline = self._build_attack_timeline(forensic_events)
        
        # Store investigation
        self.investigations[investigation_id] = investigation
        self.metrics["investigations_created"] += 1
        
        logger.info(
            f"Security investigation created",
            investigation_id=investigation_id,
            title=title,
            priority=priority.value,
            affected_agents=len(affected_agents),
            event_count=len(relevant_events)
        )
        
        return investigation
    
    async def perform_forensic_analysis(
        self,
        agent_id: Optional[uuid.UUID] = None,
        time_range_hours: int = 24,
        analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Perform forensic analysis on events.
        
        Args:
            agent_id: Specific agent to analyze (None for all)
            time_range_hours: Time range in hours
            analysis_type: Type of analysis to perform
            
        Returns:
            Forensic analysis results
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)
        
        # Filter events
        relevant_events = []
        for event in self.forensic_events:
            if event.timestamp >= cutoff_time:
                if agent_id is None or event.agent_id == agent_id:
                    relevant_events.append(event)
        
        if not relevant_events:
            return {"error": "No events found for analysis"}
        
        # Perform analysis
        analysis_result = await self.forensic_analyzer.analyze_events(
            relevant_events, analysis_type
        )
        
        # Add summary statistics
        analysis_result["summary"] = {
            "total_events": len(relevant_events),
            "unique_agents": len(set(e.agent_id for e in relevant_events if e.agent_id)),
            "unique_sessions": len(set(e.session_id for e in relevant_events if e.session_id)),
            "severity_distribution": dict(Counter(e.severity.value for e in relevant_events)),
            "avg_risk_score": np.mean([e.risk_score for e in relevant_events]),
            "high_risk_events": len([e for e in relevant_events if e.risk_score > 0.7])
        }
        
        return analysis_result
    
    async def search_events(
        self,
        query: Optional[str] = None,
        agent_id: Optional[uuid.UUID] = None,
        event_type: Optional[ForensicEventType] = None,
        severity: Optional[ThreatLevel] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        risk_score_min: Optional[float] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[ForensicEvent]:
        """
        Search forensic events with flexible criteria.
        
        Args:
            query: Text query for description/indicators
            agent_id: Specific agent ID
            event_type: Event type filter
            severity: Severity filter
            start_time: Start time filter
            end_time: End time filter
            risk_score_min: Minimum risk score
            tags: Required tags
            limit: Maximum results
            
        Returns:
            List of matching ForensicEvent objects
        """
        matching_events = []
        
        for event in self.forensic_events:
            # Apply filters
            if agent_id and event.agent_id != agent_id:
                continue
            
            if event_type and event.event_type != event_type:
                continue
            
            if severity and event.severity != severity:
                continue
            
            if start_time and event.timestamp < start_time:
                continue
            
            if end_time and event.timestamp > end_time:
                continue
            
            if risk_score_min and event.risk_score < risk_score_min:
                continue
            
            if tags and not any(tag in event.event_tags for tag in tags):
                continue
            
            # Text query
            if query:
                query_lower = query.lower()
                searchable_text = (
                    event.description.lower() + " " + 
                    " ".join(event.threat_indicators).lower()
                )
                if query_lower not in searchable_text:
                    continue
            
            matching_events.append(event)
            
            if len(matching_events) >= limit:
                break
        
        # Sort by timestamp (newest first)
        matching_events.sort(key=lambda e: e.timestamp, reverse=True)
        
        return matching_events
    
    def _generate_event_tags(
        self,
        result: SecurityAnalysisResult,
        threat_detections: List[ThreatDetection]
    ) -> List[str]:
        """Generate relevant tags for the event."""
        tags = []
        
        # Risk level tags
        tags.append(f"risk_{result.risk_level.value.lower()}")
        
        # Threat category tags
        for category in result.threat_categories:
            tags.append(f"threat_{category.value.lower()}")
        
        # Analysis result tags
        if not result.is_safe:
            tags.append("unsafe_command")
        
        if result.behavioral_anomalies:
            tags.append("behavioral_anomaly")
        
        if result.matched_signatures:
            tags.append("signature_match")
        
        # Threat detection tags
        for detection in threat_detections:
            tags.append(f"detection_{detection.threat_type.value.lower()}")
            if detection.confidence > 0.8:
                tags.append("high_confidence_detection")
        
        # Control decision tags
        tags.append(f"decision_{result.control_decision.value.lower()}")
        
        return tags
    
    async def _trigger_auto_investigation(
        self,
        trigger_event: ForensicEvent,
        threat_detections: List[ThreatDetection]
    ) -> None:
        """Trigger automatic investigation based on event."""
        
        if not trigger_event.agent_id:
            return
        
        # Check if investigation already exists for this agent
        existing_investigations = [
            inv for inv in self.investigations.values()
            if (trigger_event.agent_id in inv.affected_agents and
                inv.status in [InvestigationStatus.OPEN, InvestigationStatus.IN_PROGRESS])
        ]
        
        if existing_investigations:
            # Add event to existing investigation
            investigation = existing_investigations[0]
            investigation.related_events.append(trigger_event.id)
            investigation.updated_at = datetime.utcnow()
            trigger_event.parent_investigation = investigation.id
            return
        
        # Create new investigation
        investigation_title = f"Auto-Investigation: Agent {trigger_event.agent_id}"
        investigation_description = f"Automatically triggered by high-risk event: {trigger_event.description}"
        
        time_range_start = trigger_event.timestamp - timedelta(hours=2)
        time_range_end = trigger_event.timestamp + timedelta(hours=1)
        
        await self.create_investigation(
            title=investigation_title,
            description=investigation_description,
            priority=trigger_event.severity,
            affected_agents=[trigger_event.agent_id],
            time_range_start=time_range_start,
            time_range_end=time_range_end,
            created_by="auto_investigation_system"
        )
    
    def _extract_iocs(self, events: List[ForensicEvent]) -> List[str]:
        """Extract indicators of compromise from events."""
        iocs = set()
        
        for event in events:
            # Add threat indicators
            iocs.update(event.threat_indicators)
            
            # Add command hashes for high-risk commands
            if event.command_hash and event.risk_score > 0.6:
                iocs.add(f"command_hash:{event.command_hash}")
            
            # Add IP addresses for suspicious activity
            if event.ip_address and event.severity in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                iocs.add(f"suspicious_ip:{event.ip_address}")
        
        return list(iocs)
    
    def _build_attack_timeline(self, events: List[ForensicEvent]) -> List[Dict[str, Any]]:
        """Build attack timeline from events."""
        
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        timeline = []
        
        for event in sorted_events:
            timeline_entry = {
                "timestamp": event.timestamp.isoformat(),
                "event_id": event.id,
                "event_type": event.event_type.value,
                "severity": event.severity.value,
                "description": event.description,
                "risk_score": event.risk_score,
                "key_indicators": event.threat_indicators[:3]  # Top 3 indicators
            }
            
            # Add command information if available
            if event.command_hash:
                timeline_entry["command_hash"] = event.command_hash
            
            timeline.append(timeline_entry)
        
        return timeline
    
    async def _correlation_worker(self) -> None:
        """Background worker for event correlation."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                if self.config["enable_real_time_correlation"]:
                    await self._perform_event_correlation()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Correlation worker error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_worker(self) -> None:
        """Background worker for data cleanup."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_old_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")
                await asyncio.sleep(300)
    
    async def _perform_event_correlation(self) -> None:
        """Perform real-time event correlation."""
        correlation_cutoff = datetime.utcnow() - self.correlation_window
        
        # Find events within correlation window
        recent_events = [
            event for event in self.forensic_events
            if event.timestamp >= correlation_cutoff
        ]
        
        if len(recent_events) < 2:
            return
        
        # Perform correlation analysis
        correlation_result = await self.forensic_analyzer._perform_correlation_analysis(recent_events)
        
        # Update event correlation scores
        for correlation in correlation_result.get("correlations", []):
            if correlation.get("correlation_type") == "agent_activity":
                agent_id = correlation.get("agent_id")
                if agent_id:
                    # Update correlation scores for events from this agent
                    for event in recent_events:
                        if str(event.agent_id) == agent_id:
                            event.correlation_score = min(1.0, event.correlation_score + 0.1)
        
        self.metrics["correlations_found"] += len(correlation_result.get("correlations", []))
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old forensic data."""
        cutoff_time = datetime.utcnow() - timedelta(days=self.config["retention_days"])
        
        # Clean up old forensic events (keep in memory but mark as archived)
        archived_count = 0
        for event in list(self.forensic_events):
            if event.timestamp < cutoff_time:
                archived_count += 1
        
        # Clean up correlation cache
        for agent_id in list(self.correlation_cache.keys()):
            events = self.correlation_cache[agent_id]
            self.correlation_cache[agent_id] = [
                e for e in events if e.timestamp >= cutoff_time
            ]
        
        # Clean up old investigations
        investigation_cutoff = datetime.utcnow() - timedelta(days=self.config["max_investigation_duration_days"])
        closed_investigations = 0
        
        for investigation in self.investigations.values():
            if (investigation.status == InvestigationStatus.OPEN and
                investigation.created_at < investigation_cutoff):
                investigation.status = InvestigationStatus.CLOSED
                investigation.resolution_summary = "Auto-closed due to age"
                closed_investigations += 1
        
        if archived_count > 0 or closed_investigations > 0:
            logger.info(
                f"Cleanup completed: {archived_count} events archived, {closed_investigations} investigations closed"
            )
    
    def _update_metrics(self, processing_time_ms: float) -> None:
        """Update performance metrics."""
        self.metrics["forensic_events_processed"] += 1
        
        # Update average processing time
        current_avg = self.metrics["avg_forensic_processing_time_ms"]
        total_processed = self.metrics["forensic_events_processed"]
        self.metrics["avg_forensic_processing_time_ms"] = (
            (current_avg * (total_processed - 1) + processing_time_ms) / total_processed
        )
        
        # Update investigation resolution rate
        total_investigations = len(self.investigations)
        resolved_investigations = len([
            inv for inv in self.investigations.values()
            if inv.status in [InvestigationStatus.RESOLVED, InvestigationStatus.CLOSED]
        ])
        
        if total_investigations > 0:
            self.metrics["investigation_resolution_rate"] = resolved_investigations / total_investigations
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics."""
        return {
            "enhanced_security_audit": self.metrics.copy(),
            "base_audit_metrics": self.base_audit_system.get_security_dashboard(),
            "forensic_events_count": len(self.forensic_events),
            "active_investigations": len([
                inv for inv in self.investigations.values()
                if inv.status in [InvestigationStatus.OPEN, InvestigationStatus.IN_PROGRESS]
            ]),
            "total_investigations": len(self.investigations),
            "correlation_cache_size": sum(len(events) for events in self.correlation_cache.values()),
            "configuration": self.config.copy()
        }


# Factory function
async def create_enhanced_security_audit(
    base_audit_system: SecurityAuditSystem,
    db_session: AsyncSession,
    access_control_manager: AccessControlManager
) -> EnhancedSecurityAudit:
    """
    Create EnhancedSecurityAudit instance.
    
    Args:
        base_audit_system: Base security audit system
        db_session: Database session
        access_control_manager: Access control manager
        
    Returns:
        EnhancedSecurityAudit instance
    """
    audit_system = EnhancedSecurityAudit(base_audit_system, db_session, access_control_manager)
    await audit_system.initialize()
    return audit_system