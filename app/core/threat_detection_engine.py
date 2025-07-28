"""
Threat Detection Engine for LeanVibe Agent Hive 2.0.

This system provides advanced behavioral analysis, anomaly detection, and real-time threat
identification using machine learning and statistical analysis techniques.

Features:
- Real-time behavioral pattern analysis
- Machine learning-based anomaly detection
- Statistical deviation analysis for agent behavior
- Threat correlation and attribution
- Dynamic risk scoring and threat prioritization
- Integration with existing security infrastructure
- Comprehensive threat intelligence processing
"""

import asyncio
import uuid
import json
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque, Counter
import logging
from scipy import stats
import hashlib

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from .advanced_security_validator import ThreatCategory, SecurityAnalysisResult, CommandContext
from .enhanced_security_safeguards import (
    ControlDecision, SecurityContext, AgentBehaviorState, SecurityRiskLevel
)
from .security_audit import SecurityAuditSystem, ThreatLevel, AuditEventType, SecurityEvent
from ..models.agent import Agent
from ..models.context import Context

logger = structlog.get_logger()


class ThreatType(Enum):
    """Types of threats detected by the engine."""
    BEHAVIORAL_ANOMALY = "BEHAVIORAL_ANOMALY"
    COMMAND_PATTERN_ABUSE = "COMMAND_PATTERN_ABUSE"
    PRIVILEGE_ESCALATION = "PRIVILEGE_ESCALATION"
    DATA_EXFILTRATION = "DATA_EXFILTRATION"
    LATERAL_MOVEMENT = "LATERAL_MOVEMENT"
    PERSISTENCE_ATTEMPT = "PERSISTENCE_ATTEMPT"
    RESOURCE_ABUSE = "RESOURCE_ABUSE"
    RECONNAISSANCE = "RECONNAISSANCE"
    SOCIAL_ENGINEERING = "SOCIAL_ENGINEERING"
    ADVANCED_PERSISTENT_THREAT = "ADVANCED_PERSISTENT_THREAT"


class DetectionMethod(Enum):
    """Methods used for threat detection."""
    STATISTICAL_ANALYSIS = "STATISTICAL_ANALYSIS"
    MACHINE_LEARNING = "MACHINE_LEARNING"
    PATTERN_MATCHING = "PATTERN_MATCHING"
    BEHAVIORAL_PROFILING = "BEHAVIORAL_PROFILING"
    CORRELATION_ANALYSIS = "CORRELATION_ANALYSIS"
    THREAT_INTELLIGENCE = "THREAT_INTELLIGENCE"


@dataclass
class BehavioralMetric:
    """Individual behavioral metric for analysis."""
    name: str
    value: float
    baseline_mean: float
    baseline_std: float
    z_score: float
    percentile: float
    is_anomaly: bool
    confidence: float
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ThreatDetection:
    """Represents a detected threat."""
    id: str
    threat_type: ThreatType
    detection_method: DetectionMethod
    confidence: float
    severity: ThreatLevel
    
    # Detection details
    affected_agent_id: uuid.UUID
    session_id: Optional[uuid.UUID]
    detection_time: datetime
    
    # Threat characteristics
    indicators: List[str]
    evidence: Dict[str, Any]
    risk_score: float
    
    # Context and attribution
    command_sequence: List[str]
    behavioral_metrics: List[BehavioralMetric]
    related_detections: List[str] = field(default_factory=list)
    
    # Response recommendations
    recommended_actions: List[str] = field(default_factory=list)
    containment_required: bool = False
    investigation_required: bool = False
    
    # Metadata
    detection_engine_version: str = "1.0"
    false_positive_probability: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "threat_type": self.threat_type.value,
            "detection_method": self.detection_method.value,
            "confidence": self.confidence,
            "severity": self.severity.value,
            "affected_agent_id": str(self.affected_agent_id),
            "session_id": str(self.session_id) if self.session_id else None,
            "detection_time": self.detection_time.isoformat(),
            "indicators": self.indicators,
            "evidence": self.evidence,
            "risk_score": self.risk_score,
            "command_sequence": self.command_sequence,
            "behavioral_metrics": [
                {
                    "name": metric.name,
                    "value": metric.value,
                    "z_score": metric.z_score,
                    "is_anomaly": metric.is_anomaly,
                    "confidence": metric.confidence
                }
                for metric in self.behavioral_metrics
            ],
            "related_detections": self.related_detections,
            "recommended_actions": self.recommended_actions,
            "containment_required": self.containment_required,
            "investigation_required": self.investigation_required,
            "detection_engine_version": self.detection_engine_version,
            "false_positive_probability": self.false_positive_probability
        }


@dataclass
class AgentBehavioralProfile:
    """Comprehensive behavioral profile for an agent."""
    agent_id: uuid.UUID
    
    # Basic metrics
    total_commands: int = 0
    unique_commands: int = 0
    average_command_interval: float = 0.0
    
    # Temporal patterns
    active_hours: Set[int] = field(default_factory=set)
    daily_command_counts: deque = field(default_factory=lambda: deque(maxlen=30))
    hourly_patterns: Dict[int, float] = field(default_factory=dict)
    
    # Command patterns
    command_types: Counter = field(default_factory=Counter)
    command_complexity_scores: deque = field(default_factory=lambda: deque(maxlen=1000))
    failure_rate: float = 0.0
    
    # Resource access patterns
    accessed_resources: Set[str] = field(default_factory=set)
    resource_access_patterns: Dict[str, int] = field(default_factory=dict)
    
    # Risk indicators
    privilege_escalation_attempts: int = 0
    suspicious_command_count: int = 0
    anomaly_score_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Statistical baselines
    metrics_baselines: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # (mean, std)
    
    # Metadata
    profile_created: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    learning_period_days: int = 7
    is_baseline_established: bool = False


class StatisticalAnalyzer:
    """Statistical analysis engine for behavioral metrics."""
    
    def __init__(self, anomaly_threshold: float = 2.0):
        self.anomaly_threshold = anomaly_threshold  # Z-score threshold
        self.min_samples = 10  # Minimum samples for reliable statistics
    
    def analyze_metric(
        self,
        metric_name: str,
        current_value: float,
        historical_values: List[float]
    ) -> BehavioralMetric:
        """
        Analyze a behavioral metric for anomalies.
        
        Args:
            metric_name: Name of the metric
            current_value: Current metric value
            historical_values: Historical values for baseline
            
        Returns:
            BehavioralMetric with analysis results
        """
        if len(historical_values) < self.min_samples:
            # Not enough data for reliable analysis
            return BehavioralMetric(
                name=metric_name,
                value=current_value,
                baseline_mean=current_value,
                baseline_std=0.0,
                z_score=0.0,
                percentile=50.0,
                is_anomaly=False,
                confidence=0.1
            )
        
        # Calculate baseline statistics
        baseline_mean = np.mean(historical_values)
        baseline_std = np.std(historical_values)
        
        # Handle zero standard deviation
        if baseline_std == 0:
            z_score = 0.0 if current_value == baseline_mean else float('inf')
        else:
            z_score = (current_value - baseline_mean) / baseline_std
        
        # Calculate percentile
        percentile = stats.percentileofscore(historical_values, current_value)
        
        # Determine if anomalous
        is_anomaly = abs(z_score) > self.anomaly_threshold
        
        # Calculate confidence based on sample size and z-score
        confidence = min(1.0, len(historical_values) / 100) * min(1.0, abs(z_score) / 3.0)
        
        return BehavioralMetric(
            name=metric_name,
            value=current_value,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            z_score=z_score,
            percentile=percentile,
            is_anomaly=is_anomaly,
            confidence=confidence
        )
    
    def detect_time_series_anomalies(
        self,
        values: List[float],
        window_size: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Detect anomalies in time series data using sliding window.
        
        Args:
            values: Time series values
            window_size: Size of sliding window
            
        Returns:
            List of (index, anomaly_score) tuples
        """
        anomalies = []
        
        if len(values) < window_size:
            return anomalies
        
        for i in range(window_size, len(values)):
            window = values[i-window_size:i]
            current_value = values[i]
            
            window_mean = np.mean(window)
            window_std = np.std(window)
            
            if window_std > 0:
                z_score = abs((current_value - window_mean) / window_std)
                if z_score > self.anomaly_threshold:
                    anomalies.append((i, z_score))
        
        return anomalies


class BehavioralPatternAnalyzer:
    """Analyzes behavioral patterns for threat detection."""
    
    def __init__(self):
        self.pattern_matchers = {
            "command_escalation": self._detect_command_escalation,
            "data_exfiltration": self._detect_data_exfiltration_pattern,
            "lateral_movement": self._detect_lateral_movement,
            "persistence": self._detect_persistence_attempts,
            "reconnaissance": self._detect_reconnaissance_behavior
        }
    
    def analyze_command_sequence(
        self,
        commands: List[str],
        context: CommandContext
    ) -> List[Dict[str, Any]]:
        """
        Analyze a sequence of commands for behavioral patterns.
        
        Args:
            commands: List of commands in sequence
            context: Command execution context
            
        Returns:
            List of detected patterns with metadata
        """
        detected_patterns = []
        
        for pattern_name, matcher in self.pattern_matchers.items():
            try:
                result = matcher(commands, context)
                if result:
                    detected_patterns.append({
                        "pattern": pattern_name,
                        "confidence": result.get("confidence", 0.5),
                        "indicators": result.get("indicators", []),
                        "risk_score": result.get("risk_score", 0.5),
                        "metadata": result.get("metadata", {})
                    })
            except Exception as e:
                logger.error(f"Pattern matching error for {pattern_name}: {e}")
        
        return detected_patterns
    
    def _detect_command_escalation(
        self,
        commands: List[str],
        context: CommandContext
    ) -> Optional[Dict[str, Any]]:
        """Detect privilege escalation patterns in command sequence."""
        escalation_indicators = []
        escalation_commands = ['sudo', 'su', 'doas', 'runas']
        
        # Look for escalation attempts
        for i, cmd in enumerate(commands):
            cmd_lower = cmd.lower()
            for esc_cmd in escalation_commands:
                if esc_cmd in cmd_lower:
                    escalation_indicators.append({
                        "command_index": i,
                        "command": cmd[:100],  # Truncate for safety
                        "escalation_method": esc_cmd
                    })
        
        if not escalation_indicators:
            return None
        
        # Calculate risk based on frequency and context
        risk_score = min(1.0, len(escalation_indicators) * 0.3)
        
        # Increase risk if multiple methods used
        unique_methods = set(ind["escalation_method"] for ind in escalation_indicators)
        if len(unique_methods) > 1:
            risk_score *= 1.5
        
        # Context-based risk adjustment
        if context.trust_level < 0.3:
            risk_score *= 1.5
        
        return {
            "confidence": min(1.0, len(escalation_indicators) * 0.4),
            "indicators": escalation_indicators,
            "risk_score": min(1.0, risk_score),
            "metadata": {
                "escalation_attempts": len(escalation_indicators),
                "unique_methods": len(unique_methods),
                "trust_level": context.trust_level
            }
        }
    
    def _detect_data_exfiltration_pattern(
        self,
        commands: List[str],
        context: CommandContext
    ) -> Optional[Dict[str, Any]]:
        """Detect data exfiltration patterns."""
        exfil_indicators = []
        
        # Network transfer commands
        network_commands = ['curl', 'wget', 'scp', 'rsync', 'nc', 'netcat']
        
        # Data collection commands
        data_commands = ['find', 'grep', 'cat', 'tar', 'zip', 'gzip']
        
        network_activity = []
        data_collection = []
        
        for i, cmd in enumerate(commands):
            cmd_lower = cmd.lower()
            
            # Check for network activity
            for net_cmd in network_commands:
                if net_cmd in cmd_lower:
                    network_activity.append({"index": i, "command": cmd[:100], "method": net_cmd})
            
            # Check for data collection
            for data_cmd in data_commands:
                if data_cmd in cmd_lower:
                    data_collection.append({"index": i, "command": cmd[:100], "method": data_cmd})
        
        # Pattern: data collection followed by network transfer
        if data_collection and network_activity:
            for data_op in data_collection:
                for net_op in network_activity:
                    if net_op["index"] > data_op["index"]:
                        exfil_indicators.append({
                            "data_collection": data_op,
                            "network_transfer": net_op,
                            "time_gap": net_op["index"] - data_op["index"]
                        })
        
        if not exfil_indicators:
            return None
        
        # Calculate risk score
        risk_score = min(1.0, len(exfil_indicators) * 0.4)
        
        # Higher risk for shorter time gaps (immediate exfiltration)
        avg_time_gap = np.mean([ind["time_gap"] for ind in exfil_indicators])
        if avg_time_gap < 3:  # Commands within 3 steps
            risk_score *= 1.5
        
        return {
            "confidence": min(1.0, len(exfil_indicators) * 0.6),
            "indicators": exfil_indicators,
            "risk_score": min(1.0, risk_score),
            "metadata": {
                "data_collection_ops": len(data_collection),
                "network_transfer_ops": len(network_activity),
                "avg_time_gap": avg_time_gap
            }
        }
    
    def _detect_lateral_movement(
        self,
        commands: List[str],
        context: CommandContext
    ) -> Optional[Dict[str, Any]]:
        """Detect lateral movement patterns."""
        movement_indicators = []
        
        # Remote access and discovery commands
        lateral_commands = [
            'ssh', 'telnet', 'rsh', 'rlogin',
            'psexec', 'winrm', 'wmic',
            'arp', 'ping', 'nmap', 'netstat'
        ]
        
        for i, cmd in enumerate(commands):
            cmd_lower = cmd.lower()
            for lat_cmd in lateral_commands:
                if lat_cmd in cmd_lower:
                    movement_indicators.append({
                        "command_index": i,
                        "command": cmd[:100],
                        "method": lat_cmd
                    })
        
        if len(movement_indicators) < 2:  # Need multiple commands for pattern
            return None
        
        # Look for network discovery followed by access attempts
        discovery_commands = ['arp', 'ping', 'nmap', 'netstat']
        access_commands = ['ssh', 'telnet', 'rsh', 'rlogin', 'psexec']
        
        pattern_strength = 0
        for i, ind1 in enumerate(movement_indicators[:-1]):
            for ind2 in movement_indicators[i+1:]:
                if (ind1["method"] in discovery_commands and 
                    ind2["method"] in access_commands):
                    pattern_strength += 1
        
        risk_score = min(1.0, pattern_strength * 0.5 + len(movement_indicators) * 0.1)
        
        return {
            "confidence": min(1.0, pattern_strength * 0.4),
            "indicators": movement_indicators,
            "risk_score": risk_score,
            "metadata": {
                "movement_commands": len(movement_indicators),
                "pattern_strength": pattern_strength
            }
        }
    
    def _detect_persistence_attempts(
        self,
        commands: List[str],
        context: CommandContext
    ) -> Optional[Dict[str, Any]]:
        """Detect persistence mechanism attempts."""
        persistence_indicators = []
        
        # Persistence mechanisms
        persistence_patterns = [
            (r'crontab.*-e', 'cron_job'),
            (r'echo.*>>.*/\.bashrc', 'shell_profile'),
            (r'echo.*>>.*/\.profile', 'shell_profile'),
            (r'systemctl.*enable', 'systemd_service'),
            (r'update-rc\.d.*enable', 'init_script'),
            (r'reg.*add.*run', 'registry_run_key'),
            (r'schtasks.*create', 'scheduled_task')
        ]
        
        for i, cmd in enumerate(commands):
            for pattern, mechanism in persistence_patterns:
                if re.search(pattern, cmd, re.IGNORECASE):
                    persistence_indicators.append({
                        "command_index": i,
                        "command": cmd[:100],
                        "mechanism": mechanism,
                        "pattern": pattern
                    })
        
        if not persistence_indicators:
            return None
        
        # Higher risk for multiple persistence mechanisms
        unique_mechanisms = set(ind["mechanism"] for ind in persistence_indicators)
        risk_score = min(1.0, len(unique_mechanisms) * 0.4 + len(persistence_indicators) * 0.2)
        
        return {
            "confidence": min(1.0, len(persistence_indicators) * 0.5),
            "indicators": persistence_indicators,
            "risk_score": risk_score,
            "metadata": {
                "persistence_attempts": len(persistence_indicators),
                "unique_mechanisms": len(unique_mechanisms)
            }
        }
    
    def _detect_reconnaissance_behavior(
        self,
        commands: List[str],
        context: CommandContext
    ) -> Optional[Dict[str, Any]]:
        """Detect reconnaissance behavior patterns."""
        recon_indicators = []
        
        # System information gathering commands
        recon_commands = [
            ('whoami', 'user_enumeration'),
            ('id', 'user_enumeration'),
            ('uname', 'system_info'),
            ('hostname', 'system_info'),
            ('ps', 'process_enumeration'),
            ('netstat', 'network_enumeration'),
            ('ifconfig', 'network_info'),
            ('mount', 'filesystem_info'),
            ('df', 'filesystem_info'),
            ('ls -la /', 'filesystem_enumeration'),
            ('cat /etc/passwd', 'user_enumeration'),
            ('cat /etc/group', 'user_enumeration')
        ]
        
        for i, cmd in enumerate(commands):
            cmd_lower = cmd.lower()
            for recon_cmd, category in recon_commands:
                if recon_cmd in cmd_lower:
                    recon_indicators.append({
                        "command_index": i,
                        "command": cmd[:100],
                        "category": category,
                        "recon_type": recon_cmd
                    })
        
        if len(recon_indicators) < 3:  # Need multiple recon commands
            return None
        
        # Calculate diversity of reconnaissance
        unique_categories = set(ind["category"] for ind in recon_indicators)
        diversity_score = len(unique_categories) / 5.0  # Max 5 categories
        
        # Risk increases with diversity and volume
        risk_score = min(1.0, diversity_score * 0.6 + len(recon_indicators) * 0.1)
        
        return {
            "confidence": min(1.0, len(recon_indicators) * 0.3),
            "indicators": recon_indicators,
            "risk_score": risk_score,
            "metadata": {
                "recon_commands": len(recon_indicators),
                "categories_explored": len(unique_categories),
                "diversity_score": diversity_score
            }
        }


class ThreatDetectionEngine:
    """
    Advanced Threat Detection Engine for behavioral analysis and threat identification.
    
    Provides comprehensive threat detection using statistical analysis, behavioral profiling,
    and pattern recognition to identify sophisticated attacks and anomalous behavior.
    """
    
    def __init__(
        self,
        audit_system: SecurityAuditSystem,
        statistical_analyzer: Optional[StatisticalAnalyzer] = None,
        pattern_analyzer: Optional[BehavioralPatternAnalyzer] = None
    ):
        self.audit_system = audit_system
        self.statistical_analyzer = statistical_analyzer or StatisticalAnalyzer()
        self.pattern_analyzer = pattern_analyzer or BehavioralPatternAnalyzer()
        
        # Agent profiles and state
        self.agent_profiles: Dict[uuid.UUID, AgentBehavioralProfile] = {}
        self.command_history: Dict[uuid.UUID, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Detection history and correlation
        self.detection_history: deque = deque(maxlen=10000)
        self.correlation_cache: Dict[str, List[ThreatDetection]] = defaultdict(list)
        
        # Performance metrics
        self.metrics = {
            "detections_performed": 0,
            "threats_detected": 0,
            "behavioral_anomalies": 0,
            "pattern_matches": 0,
            "false_positives": 0,
            "avg_detection_time_ms": 0.0,
            "profiles_maintained": 0
        }
        
        # Configuration
        self.config = {
            "enable_statistical_analysis": True,
            "enable_pattern_analysis": True,
            "enable_correlation_analysis": True,
            "min_profile_age_hours": 24,
            "max_detection_time_ms": 100,
            "correlation_window_hours": 6,
            "auto_tune_thresholds": True,
            "learning_rate": 0.1,
            "anomaly_sensitivity": 0.7
        }
        
        # Initialize background tasks
        self._correlation_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize the threat detection engine."""
        # Start background tasks
        self._correlation_task = asyncio.create_task(self._correlation_worker())
        self._cleanup_task = asyncio.create_task(self._cleanup_worker())
        
        logger.info("Threat Detection Engine initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the threat detection engine."""
        if self._correlation_task:
            self._correlation_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        logger.info("Threat Detection Engine shutdown completed")
    
    async def analyze_agent_behavior(
        self,
        agent_id: uuid.UUID,
        command: str,
        context: CommandContext,
        security_result: SecurityAnalysisResult
    ) -> List[ThreatDetection]:
        """
        Analyze agent behavior for threats and anomalies.
        
        Args:
            agent_id: Agent identifier
            command: Executed command
            context: Command context
            security_result: Previous security analysis result
            
        Returns:
            List of detected threats
        """
        start_time = time.time()
        detected_threats = []
        
        try:
            # Update agent profile
            profile = await self._update_agent_profile(agent_id, command, context)
            
            # Add command to history
            self.command_history[agent_id].append({
                "command": command,
                "timestamp": context.timestamp,
                "context": context,
                "security_result": security_result
            })
            
            # Statistical analysis
            if self.config["enable_statistical_analysis"]:
                statistical_threats = await self._perform_statistical_analysis(
                    agent_id, profile, context
                )
                detected_threats.extend(statistical_threats)
            
            # Pattern analysis
            if self.config["enable_pattern_analysis"]:
                pattern_threats = await self._perform_pattern_analysis(
                    agent_id, command, context
                )
                detected_threats.extend(pattern_threats)
            
            # Behavioral profiling
            behavioral_threats = await self._perform_behavioral_profiling(
                agent_id, profile, context
            )
            detected_threats.extend(behavioral_threats)
            
            # Update detection history
            for threat in detected_threats:
                self.detection_history.append(threat)
                self.correlation_cache[str(agent_id)].append(threat)
            
            # Update metrics
            detection_time = (time.time() - start_time) * 1000
            self._update_metrics(detected_threats, detection_time)
            
            # Log significant detections
            if detected_threats:
                await self._log_threat_detections(agent_id, detected_threats, context)
            
            return detected_threats
            
        except Exception as e:
            logger.error(f"Threat detection analysis error: {e}")
            return []
    
    async def _update_agent_profile(
        self,
        agent_id: uuid.UUID,
        command: str,
        context: CommandContext
    ) -> AgentBehavioralProfile:
        """Update behavioral profile for agent."""
        
        if agent_id not in self.agent_profiles:
            self.agent_profiles[agent_id] = AgentBehavioralProfile(agent_id=agent_id)
        
        profile = self.agent_profiles[agent_id]
        
        # Update basic metrics
        profile.total_commands += 1
        profile.unique_commands = len(set(
            entry["command"] for entry in self.command_history[agent_id]
        ))
        
        # Update temporal patterns
        current_hour = context.timestamp.hour
        profile.active_hours.add(current_hour)
        
        if current_hour not in profile.hourly_patterns:
            profile.hourly_patterns[current_hour] = 0
        profile.hourly_patterns[current_hour] += 1
        
        # Update command interval
        if len(self.command_history[agent_id]) > 1:
            last_timestamp = list(self.command_history[agent_id])[-2]["timestamp"]
            interval = (context.timestamp - last_timestamp).total_seconds()
            
            # Update rolling average
            if profile.average_command_interval == 0:
                profile.average_command_interval = interval
            else:
                profile.average_command_interval = (
                    profile.average_command_interval * 0.9 + interval * 0.1
                )
        
        # Analyze command complexity
        complexity_score = self._calculate_command_complexity(command)
        profile.command_complexity_scores.append(complexity_score)
        
        # Update command type distribution
        intent = self._extract_command_intent(command)
        profile.command_types[intent] += 1
        
        # Update resource access patterns
        resources = self._extract_accessed_resources(command)
        profile.accessed_resources.update(resources)
        for resource in resources:
            profile.resource_access_patterns[resource] = (
                profile.resource_access_patterns.get(resource, 0) + 1
            )
        
        # Check if baseline is established
        profile_age = datetime.utcnow() - profile.profile_created
        if (profile_age.total_seconds() > self.config["min_profile_age_hours"] * 3600 and
            profile.total_commands > 100):
            profile.is_baseline_established = True
        
        profile.last_updated = datetime.utcnow()
        return profile
    
    async def _perform_statistical_analysis(
        self,
        agent_id: uuid.UUID,
        profile: AgentBehavioralProfile,
        context: CommandContext
    ) -> List[ThreatDetection]:
        """Perform statistical analysis for anomaly detection."""
        
        if not profile.is_baseline_established:
            return []
        
        threats = []
        behavioral_metrics = []
        
        # Analyze command frequency
        current_frequency = 1.0 / max(profile.average_command_interval, 1.0)  # Commands per second
        frequency_history = [
            1.0 / max(entry.get("interval", 60), 1.0) 
            for entry in profile.daily_command_counts
        ]
        
        if frequency_history:
            freq_metric = self.statistical_analyzer.analyze_metric(
                "command_frequency", current_frequency, frequency_history
            )
            behavioral_metrics.append(freq_metric)
            
            if freq_metric.is_anomaly and freq_metric.confidence > 0.7:
                threats.append(ThreatDetection(
                    id=str(uuid.uuid4()),
                    threat_type=ThreatType.BEHAVIORAL_ANOMALY,
                    detection_method=DetectionMethod.STATISTICAL_ANALYSIS,
                    confidence=freq_metric.confidence,
                    severity=ThreatLevel.MEDIUM if freq_metric.z_score > 3 else ThreatLevel.LOW,
                    affected_agent_id=agent_id,
                    session_id=context.session_id,
                    detection_time=context.timestamp,
                    indicators=[f"Unusual command frequency: {freq_metric.z_score:.2f} std deviations"],
                    evidence={"frequency_metric": freq_metric.__dict__},
                    risk_score=min(1.0, abs(freq_metric.z_score) / 5.0),
                    command_sequence=[],
                    behavioral_metrics=[freq_metric]
                ))
        
        # Analyze complexity patterns
        if len(profile.command_complexity_scores) > 10:
            current_complexity = profile.command_complexity_scores[-1]
            complexity_history = list(profile.command_complexity_scores)[:-1]
            
            complexity_metric = self.statistical_analyzer.analyze_metric(
                "command_complexity", current_complexity, complexity_history
            )
            behavioral_metrics.append(complexity_metric)
            
            if complexity_metric.is_anomaly and complexity_metric.confidence > 0.6:
                threats.append(ThreatDetection(
                    id=str(uuid.uuid4()),
                    threat_type=ThreatType.BEHAVIORAL_ANOMALY,
                    detection_method=DetectionMethod.STATISTICAL_ANALYSIS,
                    confidence=complexity_metric.confidence,
                    severity=ThreatLevel.LOW,
                    affected_agent_id=agent_id,
                    session_id=context.session_id,
                    detection_time=context.timestamp,
                    indicators=[f"Unusual command complexity: {complexity_metric.z_score:.2f} std deviations"],
                    evidence={"complexity_metric": complexity_metric.__dict__},
                    risk_score=min(1.0, abs(complexity_metric.z_score) / 4.0),
                    command_sequence=[],
                    behavioral_metrics=[complexity_metric]
                ))
        
        # Analyze temporal patterns
        if len(profile.active_hours) > 0:
            typical_hours = set(profile.hourly_patterns.keys())
            current_hour = context.timestamp.hour
            
            if current_hour not in typical_hours and profile.total_commands > 50:
                threats.append(ThreatDetection(
                    id=str(uuid.uuid4()),
                    threat_type=ThreatType.BEHAVIORAL_ANOMALY,
                    detection_method=DetectionMethod.BEHAVIORAL_PROFILING,
                    confidence=0.6,
                    severity=ThreatLevel.LOW,
                    affected_agent_id=agent_id,
                    session_id=context.session_id,
                    detection_time=context.timestamp,
                    indicators=[f"Activity outside typical hours: {current_hour}:00"],
                    evidence={
                        "current_hour": current_hour,
                        "typical_hours": list(typical_hours)
                    },
                    risk_score=0.4,
                    command_sequence=[],
                    behavioral_metrics=behavioral_metrics
                ))
        
        return threats
    
    async def _perform_pattern_analysis(
        self,
        agent_id: uuid.UUID,
        command: str,
        context: CommandContext
    ) -> List[ThreatDetection]:
        """Perform pattern analysis on command sequences."""
        
        # Get recent command history
        recent_commands = []
        if agent_id in self.command_history:
            recent_entries = list(self.command_history[agent_id])[-20:]  # Last 20 commands
            recent_commands = [entry["command"] for entry in recent_entries]
        
        if len(recent_commands) < 3:
            return []
        
        # Analyze patterns
        detected_patterns = self.pattern_analyzer.analyze_command_sequence(
            recent_commands, context
        )
        
        threats = []
        for pattern in detected_patterns:
            if pattern["confidence"] > 0.5:
                # Map pattern to threat type
                threat_type_map = {
                    "command_escalation": ThreatType.PRIVILEGE_ESCALATION,
                    "data_exfiltration": ThreatType.DATA_EXFILTRATION,
                    "lateral_movement": ThreatType.LATERAL_MOVEMENT,
                    "persistence": ThreatType.PERSISTENCE_ATTEMPT,
                    "reconnaissance": ThreatType.RECONNAISSANCE
                }
                
                threat_type = threat_type_map.get(
                    pattern["pattern"], ThreatType.COMMAND_PATTERN_ABUSE
                )
                
                # Determine severity based on risk score
                if pattern["risk_score"] > 0.8:
                    severity = ThreatLevel.HIGH
                elif pattern["risk_score"] > 0.6:
                    severity = ThreatLevel.MEDIUM
                else:
                    severity = ThreatLevel.LOW
                
                threat = ThreatDetection(
                    id=str(uuid.uuid4()),
                    threat_type=threat_type,
                    detection_method=DetectionMethod.PATTERN_MATCHING,
                    confidence=pattern["confidence"],
                    severity=severity,
                    affected_agent_id=agent_id,
                    session_id=context.session_id,
                    detection_time=context.timestamp,
                    indicators=pattern["indicators"],
                    evidence=pattern["metadata"],
                    risk_score=pattern["risk_score"],
                    command_sequence=recent_commands[-10:],  # Last 10 commands
                    behavioral_metrics=[]
                )
                
                # Add specific recommendations based on pattern
                if pattern["pattern"] == "command_escalation":
                    threat.recommended_actions = [
                        "Review privilege escalation attempts",
                        "Verify agent authorization level",
                        "Monitor for unauthorized access"
                    ]
                    threat.investigation_required = True
                elif pattern["pattern"] == "data_exfiltration":
                    threat.recommended_actions = [
                        "Monitor network traffic",
                        "Review data access patterns",
                        "Check for unauthorized data transfers"
                    ]
                    threat.containment_required = True
                
                threats.append(threat)
        
        return threats
    
    async def _perform_behavioral_profiling(
        self,
        agent_id: uuid.UUID,
        profile: AgentBehavioralProfile,
        context: CommandContext
    ) -> List[ThreatDetection]:
        """Perform behavioral profiling analysis."""
        
        threats = []
        
        # Check for rapid privilege escalation attempts
        if profile.privilege_escalation_attempts > 0:
            recent_escalations = 0
            cutoff_time = context.timestamp - timedelta(hours=1)
            
            for entry in self.command_history[agent_id]:
                if entry["timestamp"] >= cutoff_time:
                    if any(esc in entry["command"].lower() for esc in ['sudo', 'su', 'doas']):
                        recent_escalations += 1
            
            if recent_escalations > 5:  # More than 5 escalations in an hour
                threats.append(ThreatDetection(
                    id=str(uuid.uuid4()),
                    threat_type=ThreatType.PRIVILEGE_ESCALATION,
                    detection_method=DetectionMethod.BEHAVIORAL_PROFILING,
                    confidence=0.8,
                    severity=ThreatLevel.HIGH,
                    affected_agent_id=agent_id,
                    session_id=context.session_id,
                    detection_time=context.timestamp,
                    indicators=[f"Rapid privilege escalation attempts: {recent_escalations} in 1 hour"],
                    evidence={"escalation_count": recent_escalations, "time_window": "1 hour"},
                    risk_score=0.9,
                    command_sequence=[],
                    behavioral_metrics=[],
                    recommended_actions=["Immediate review of agent permissions"],
                    investigation_required=True
                ))
        
        # Check for resource abuse patterns
        unique_resources_accessed = len(profile.accessed_resources)
        if unique_resources_accessed > 100:  # Accessing many different resources
            threats.append(ThreatDetection(
                id=str(uuid.uuid4()),
                threat_type=ThreatType.RESOURCE_ABUSE,
                detection_method=DetectionMethod.BEHAVIORAL_PROFILING,
                confidence=0.7,
                severity=ThreatLevel.MEDIUM,
                affected_agent_id=agent_id,
                session_id=context.session_id,
                detection_time=context.timestamp,
                indicators=[f"Excessive resource access: {unique_resources_accessed} unique resources"],
                evidence={"unique_resources": unique_resources_accessed},
                risk_score=min(1.0, unique_resources_accessed / 200),
                command_sequence=[],
                behavioral_metrics=[],
                recommended_actions=["Review resource access patterns"],
                investigation_required=False
            ))
        
        return threats
    
    def _calculate_command_complexity(self, command: str) -> float:
        """Calculate complexity score for a command."""
        complexity_score = 0.0
        
        # Base complexity from command length
        complexity_score += min(len(command) / 100, 1.0) * 0.2
        
        # Pipe operations
        complexity_score += command.count('|') * 0.1
        
        # Redirection operations
        complexity_score += (command.count('>') + command.count('<')) * 0.1
        
        # Command chaining
        complexity_score += (command.count('&&') + command.count('||') + command.count(';')) * 0.15
        
        # Regular expressions
        if re.search(r'grep.*-[EP]', command) or 'sed' in command or 'awk' in command:
            complexity_score += 0.3
        
        # Variable usage
        complexity_score += len(re.findall(r'\$\w+', command)) * 0.05
        
        # Function calls or code execution
        if any(pattern in command for pattern in ['eval', 'exec', 'python -c', 'perl -e']):
            complexity_score += 0.4
        
        return min(complexity_score, 1.0)
    
    def _extract_command_intent(self, command: str) -> str:
        """Extract primary intent from command."""
        # Simplified intent extraction
        if any(cmd in command.lower() for cmd in ['ls', 'dir', 'find', 'cat', 'grep']):
            return "information_gathering"
        elif any(cmd in command.lower() for cmd in ['rm', 'del', 'rmdir', 'unlink']):
            return "file_deletion"
        elif any(cmd in command.lower() for cmd in ['cp', 'mv', 'copy', 'move']):
            return "file_manipulation"
        elif any(cmd in command.lower() for cmd in ['wget', 'curl', 'scp', 'rsync']):
            return "network_transfer"
        elif any(cmd in command.lower() for cmd in ['sudo', 'su', 'doas']):
            return "privilege_escalation"
        elif any(cmd in command.lower() for cmd in ['python', 'node', 'java', 'gcc']):
            return "code_execution"
        else:
            return "other"
    
    def _extract_accessed_resources(self, command: str) -> Set[str]:
        """Extract resources accessed by command."""
        resources = set()
        
        # File paths
        file_patterns = [
            r'/[a-zA-Z0-9/_.-]+',
            r'\./[a-zA-Z0-9/_.-]+',
            r'~/[a-zA-Z0-9/_.-]+'
        ]
        
        for pattern in file_patterns:
            matches = re.findall(pattern, command)
            resources.update(matches)
        
        # Network resources
        network_patterns = [
            r'https?://[^\s]+',
            r'ftp://[^\s]+',
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        ]
        
        for pattern in network_patterns:
            matches = re.findall(pattern, command)
            resources.update(matches)
        
        return resources
    
    async def _correlation_worker(self) -> None:
        """Background worker for threat correlation analysis."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._perform_correlation_analysis()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Correlation worker error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _cleanup_worker(self) -> None:
        """Background worker for cleaning up old data."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_old_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")
                await asyncio.sleep(300)  # Wait before retrying
    
    async def _perform_correlation_analysis(self) -> None:
        """Perform correlation analysis on recent detections."""
        correlation_window = timedelta(hours=self.config["correlation_window_hours"])
        cutoff_time = datetime.utcnow() - correlation_window
        
        # Get recent detections
        recent_detections = [
            detection for detection in self.detection_history
            if detection.detection_time >= cutoff_time
        ]
        
        if len(recent_detections) < 2:
            return
        
        # Group by agent
        detections_by_agent = defaultdict(list)
        for detection in recent_detections:
            detections_by_agent[detection.affected_agent_id].append(detection)
        
        # Look for correlated threats
        for agent_id, detections in detections_by_agent.items():
            if len(detections) > 3:  # Multiple detections for same agent
                threat_types = [d.threat_type for d in detections]
                
                # Check for APT-like behavior (multiple threat types)
                if len(set(threat_types)) >= 3:
                    await self._create_apt_detection(agent_id, detections)
    
    async def _create_apt_detection(
        self,
        agent_id: uuid.UUID,
        related_detections: List[ThreatDetection]
    ) -> None:
        """Create APT (Advanced Persistent Threat) detection."""
        
        apt_detection = ThreatDetection(
            id=str(uuid.uuid4()),
            threat_type=ThreatType.ADVANCED_PERSISTENT_THREAT,
            detection_method=DetectionMethod.CORRELATION_ANALYSIS,
            confidence=0.9,
            severity=ThreatLevel.CRITICAL,
            affected_agent_id=agent_id,
            session_id=None,
            detection_time=datetime.utcnow(),
            indicators=[
                f"Multiple threat types detected: {len(set(d.threat_type for d in related_detections))}",
                f"Total detections: {len(related_detections)}"
            ],
            evidence={
                "related_detection_ids": [d.id for d in related_detections],
                "threat_types": list(set(d.threat_type.value for d in related_detections)),
                "time_span_hours": (max(d.detection_time for d in related_detections) - 
                                  min(d.detection_time for d in related_detections)).total_seconds() / 3600
            },
            risk_score=1.0,
            command_sequence=[],
            behavioral_metrics=[],
            related_detections=[d.id for d in related_detections],
            recommended_actions=[
                "Immediate containment required",
                "Full forensic investigation",
                "Review all agent activities",
                "Check for lateral movement"
            ],
            containment_required=True,
            investigation_required=True
        )
        
        self.detection_history.append(apt_detection)
        
        # Log critical APT detection
        logger.critical(
            "Advanced Persistent Threat detected",
            agent_id=str(agent_id),
            detection_id=apt_detection.id,
            related_detections=len(related_detections)
        )
        
        # Create security event for immediate attention
        security_event = SecurityEvent(
            id=uuid.uuid4(),
            event_type=AuditEventType.SUSPICIOUS_PATTERN,
            threat_level=ThreatLevel.CRITICAL,
            agent_id=agent_id,
            context_id=None,
            session_id=None,
            description="Advanced Persistent Threat detected through correlation analysis",
            details=apt_detection.to_dict(),
            timestamp=datetime.utcnow()
        )
        
        await self.audit_system._handle_security_event(security_event)
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old behavioral data and detections."""
        cutoff_time = datetime.utcnow() - timedelta(days=30)
        
        # Clean up old command history
        for agent_id in list(self.command_history.keys()):
            history = self.command_history[agent_id]
            # Remove entries older than 30 days
            while history and history[0]["timestamp"] < cutoff_time:
                history.popleft()
        
        # Clean up old correlation cache
        for agent_id in list(self.correlation_cache.keys()):
            detections = self.correlation_cache[agent_id]
            self.correlation_cache[agent_id] = [
                d for d in detections if d.detection_time >= cutoff_time
            ]
        
        # Update profiles maintenance count
        self.metrics["profiles_maintained"] = len(self.agent_profiles)
    
    async def _log_threat_detections(
        self,
        agent_id: uuid.UUID,
        detections: List[ThreatDetection],
        context: CommandContext
    ) -> None:
        """Log threat detections for audit and monitoring."""
        
        for detection in detections:
            if detection.severity in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                logger.warning(
                    "High-severity threat detected",
                    agent_id=str(agent_id),
                    detection_id=detection.id,
                    threat_type=detection.threat_type.value,
                    confidence=detection.confidence,
                    severity=detection.severity.value
                )
            
            # Create security event for significant detections
            if detection.confidence > 0.7:
                security_event = SecurityEvent(
                    id=uuid.uuid4(),
                    event_type=AuditEventType.SUSPICIOUS_PATTERN,
                    threat_level=detection.severity,
                    agent_id=agent_id,
                    context_id=None,
                    session_id=context.session_id,
                    description=f"Threat detected: {detection.threat_type.value}",
                    details=detection.to_dict(),
                    timestamp=detection.detection_time
                )
                
                await self.audit_system._handle_security_event(security_event)
    
    def _update_metrics(self, detections: List[ThreatDetection], detection_time: float) -> None:
        """Update engine performance metrics."""
        self.metrics["detections_performed"] += 1
        
        if detections:
            self.metrics["threats_detected"] += len(detections)
            
            # Count by detection method
            for detection in detections:
                if detection.detection_method == DetectionMethod.STATISTICAL_ANALYSIS:
                    self.metrics["behavioral_anomalies"] += 1
                elif detection.detection_method == DetectionMethod.PATTERN_MATCHING:
                    self.metrics["pattern_matches"] += 1
        
        # Update average detection time
        current_avg = self.metrics["avg_detection_time_ms"]
        total_detections = self.metrics["detections_performed"]
        self.metrics["avg_detection_time_ms"] = (
            (current_avg * (total_detections - 1) + detection_time) / total_detections
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive engine metrics."""
        return {
            "threat_detection_engine": self.metrics.copy(),
            "agent_profiles": len(self.agent_profiles),
            "detection_history_size": len(self.detection_history),
            "correlation_cache_size": sum(len(detections) for detections in self.correlation_cache.values()),
            "configuration": self.config.copy()
        }
    
    def get_agent_threat_summary(self, agent_id: uuid.UUID) -> Dict[str, Any]:
        """Get threat summary for specific agent."""
        
        if agent_id not in self.agent_profiles:
            return {"error": "Agent profile not found"}
        
        profile = self.agent_profiles[agent_id]
        
        # Get recent detections for agent
        recent_detections = [
            d for d in self.detection_history
            if d.affected_agent_id == agent_id and 
            d.detection_time >= datetime.utcnow() - timedelta(days=7)
        ]
        
        # Summarize threat types
        threat_type_counts = Counter(d.threat_type.value for d in recent_detections)
        
        return {
            "agent_id": str(agent_id),
            "profile_age_days": (datetime.utcnow() - profile.profile_created).days,
            "baseline_established": profile.is_baseline_established,
            "total_commands": profile.total_commands,
            "recent_detections": len(recent_detections),
            "threat_type_distribution": dict(threat_type_counts),
            "risk_indicators": {
                "privilege_escalation_attempts": profile.privilege_escalation_attempts,
                "suspicious_command_count": profile.suspicious_command_count,
                "unique_resources_accessed": len(profile.accessed_resources)
            },
            "behavioral_summary": {
                "average_command_interval": profile.average_command_interval,
                "active_hours": len(profile.active_hours),
                "command_type_diversity": len(profile.command_types)
            }
        }


# Factory function
async def create_threat_detection_engine(
    audit_system: SecurityAuditSystem
) -> ThreatDetectionEngine:
    """
    Create ThreatDetectionEngine instance.
    
    Args:
        audit_system: Security audit system
        
    Returns:
        ThreatDetectionEngine instance
    """
    engine = ThreatDetectionEngine(audit_system)
    await engine.initialize()
    return engine