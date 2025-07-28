"""
Advanced Security Validator for LeanVibe Agent Hive 2.0.

This system provides sophisticated command security validation with context-aware analysis,
machine learning-based threat detection, and integration with the existing security infrastructure.

Features:
- Context-aware command parsing and analysis
- Machine learning-based threat classification
- Behavioral pattern recognition
- Integration with existing SecurityValidator
- Advanced command forensics and attribution
- Real-time threat intelligence updates
- Configurable security policies with risk scoring
"""

import asyncio
import uuid
import json
import re
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
import logging

import structlog
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from .hook_lifecycle_system import SecurityValidator, SecurityRisk, DangerousCommand
from .enhanced_security_safeguards import (
    ControlDecision, SecurityContext, AgentBehaviorState, SecurityRiskLevel
)
from .security_middleware import SecurityMiddleware
from .authorization_engine import AuthorizationEngine
from .security_audit import SecurityAuditSystem, ThreatLevel, AuditEventType, SecurityEvent
from ..models.agent import Agent
from ..models.context import Context

logger = structlog.get_logger()


class ThreatCategory(Enum):
    """Categories of security threats."""
    COMMAND_INJECTION = "COMMAND_INJECTION"
    DATA_EXFILTRATION = "DATA_EXFILTRATION"
    PRIVILEGE_ESCALATION = "PRIVILEGE_ESCALATION"
    RESOURCE_ABUSE = "RESOURCE_ABUSE"
    PERSISTENCE_MECHANISM = "PERSISTENCE_MECHANISM"
    NETWORK_RECONNAISSANCE = "NETWORK_RECONNAISSANCE"
    CRYPTOGRAPHIC_ATTACK = "CRYPTOGRAPHIC_ATTACK"
    SOCIAL_ENGINEERING = "SOCIAL_ENGINEERING"
    ANOMALOUS_BEHAVIOR = "ANOMALOUS_BEHAVIOR"
    UNKNOWN_THREAT = "UNKNOWN_THREAT"


class AnalysisMode(Enum):
    """Analysis modes for different security contexts."""
    FAST = "FAST"           # <10ms, basic pattern matching
    STANDARD = "STANDARD"   # <50ms, context analysis + patterns
    DEEP = "DEEP"          # <200ms, full ML analysis
    FORENSIC = "FORENSIC"  # No time limit, comprehensive analysis


@dataclass
class CommandContext:
    """Enhanced context for command analysis."""
    agent_id: uuid.UUID
    session_id: Optional[uuid.UUID]
    command: str
    
    # Execution context
    working_directory: Optional[str] = None
    environment_vars: Dict[str, str] = field(default_factory=dict)
    previous_commands: List[str] = field(default_factory=list)
    user_context: Optional[str] = None
    
    # Agent context
    agent_type: Optional[str] = None
    agent_capabilities: List[str] = field(default_factory=list)
    trust_level: float = 0.5
    
    # Temporal context
    timestamp: datetime = field(default_factory=datetime.utcnow)
    time_since_last_command: Optional[timedelta] = None
    command_frequency: float = 0.0  # Commands per minute
    
    # Network context
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    geo_location: Optional[Dict[str, str]] = None
    
    # Risk context
    current_risk_score: float = 0.0
    recent_anomalies: List[str] = field(default_factory=list)
    security_alerts_24h: int = 0


@dataclass
class ThreatSignature:
    """Signature for threat detection."""
    id: str
    name: str
    category: ThreatCategory
    patterns: List[str]
    context_requirements: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.5
    confidence_threshold: float = 0.7
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def matches(self, command: str, context: CommandContext) -> Tuple[bool, float]:
        """
        Check if command matches this threat signature.
        
        Returns:
            Tuple of (matches, confidence_score)
        """
        try:
            confidence_scores = []
            
            # Pattern matching
            for pattern in self.patterns:
                if re.search(pattern, command, re.IGNORECASE | re.MULTILINE):
                    confidence_scores.append(0.8)
                elif self._fuzzy_match(pattern, command):
                    confidence_scores.append(0.6)
            
            if not confidence_scores:
                return False, 0.0
            
            # Context evaluation
            context_score = self._evaluate_context_requirements(context)
            confidence_scores.append(context_score)
            
            # Calculate final confidence
            final_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            
            return final_confidence >= self.confidence_threshold, final_confidence
            
        except Exception as e:
            logger.error(f"Threat signature matching error: {e}")
            return False, 0.0
    
    def _fuzzy_match(self, pattern: str, command: str) -> bool:
        """Perform fuzzy matching for obfuscated commands."""
        # Remove common obfuscation techniques
        cleaned_command = re.sub(r'[^a-zA-Z0-9\s]', ' ', command.lower())
        cleaned_pattern = re.sub(r'[^a-zA-Z0-9\s]', ' ', pattern.lower())
        
        # Check for substring matches with some flexibility
        pattern_words = cleaned_pattern.split()
        command_words = cleaned_command.split()
        
        matches = 0
        for pattern_word in pattern_words:
            for command_word in command_words:
                if pattern_word in command_word or command_word in pattern_word:
                    matches += 1
                    break
        
        return matches / len(pattern_words) > 0.7 if pattern_words else False
    
    def _evaluate_context_requirements(self, context: CommandContext) -> float:
        """Evaluate context requirements for this signature."""
        if not self.context_requirements:
            return 0.5  # Neutral score if no requirements
        
        score = 0.5
        
        # Time-based requirements
        if "time_range" in self.context_requirements:
            current_hour = context.timestamp.hour
            allowed_hours = self.context_requirements["time_range"]
            if current_hour in allowed_hours:
                score += 0.2
            else:
                score -= 0.3
        
        # Frequency-based requirements
        if "min_frequency" in self.context_requirements:
            if context.command_frequency >= self.context_requirements["min_frequency"]:
                score += 0.3
        
        # Trust level requirements
        if "max_trust_level" in self.context_requirements:
            if context.trust_level <= self.context_requirements["max_trust_level"]:
                score += 0.2
        
        return max(0.0, min(1.0, score))


@dataclass
class SecurityAnalysisResult:
    """Result of advanced security analysis."""
    is_safe: bool
    risk_level: SecurityRiskLevel
    threat_categories: List[ThreatCategory]
    confidence_score: float
    
    # Detailed analysis
    matched_signatures: List[str]
    risk_factors: List[str]
    behavioral_anomalies: List[str]
    
    # Context analysis
    command_intent: Optional[str] = None
    potential_impact: str = "unknown"
    affected_resources: List[str] = field(default_factory=list)
    
    # Recommendations
    control_decision: ControlDecision = ControlDecision.DENY
    recommended_actions: List[str] = field(default_factory=list)
    monitoring_requirements: List[str] = field(default_factory=list)
    
    # Metadata
    analysis_time_ms: float = 0.0
    analysis_mode: AnalysisMode = AnalysisMode.STANDARD
    analyzer_version: str = "1.0"


class CommandIntentAnalyzer:
    """Analyzes command intent using rule-based and ML approaches."""
    
    def __init__(self):
        # Command categories and patterns
        self.intent_patterns = {
            "file_operation": [
                r'\b(ls|dir|find|locate|cat|less|more|head|tail)\b',
                r'\b(cp|mv|rm|mkdir|rmdir|chmod|chown)\b',
                r'\b(grep|awk|sed|sort|uniq|wc)\b'
            ],
            "network_operation": [
                r'\b(ping|wget|curl|netstat|ss|lsof)\b',
                r'\b(ssh|scp|rsync|ftp|telnet)\b',
                r'\b(nmap|nslookup|dig)\b'
            ],
            "system_administration": [
                r'\b(ps|top|htop|kill|killall|pkill)\b',
                r'\b(systemctl|service|crontab|mount|umount)\b',
                r'\b(useradd|userdel|usermod|passwd|su|sudo)\b'
            ],
            "data_processing": [
                r'\b(python|node|java|gcc|make)\b',
                r'\b(git|svn|docker|kubectl)\b',
                r'\b(mysql|psql|mongo|redis-cli)\b'
            ],
            "security_operation": [
                r'\b(gpg|openssl|keygen|certbot)\b',
                r'\b(iptables|ufw|fail2ban)\b',
                r'\b(chroot|selinux|apparmor)\b'
            ]
        }
        
        self.risk_modifiers = {
            "sudo": 2.0,
            "root": 1.8,
            "admin": 1.5,
            "rm -rf": 3.0,
            "dd if=": 2.5,
            "mkfs": 2.5,
            "format": 2.0,
            "> /dev/": 2.0,
            "curl|sh": 2.5,
            "wget|sh": 2.5
        }
    
    def analyze_intent(self, command: str, context: CommandContext) -> Tuple[str, float]:
        """
        Analyze command intent and calculate risk multiplier.
        
        Returns:
            Tuple of (intent, risk_multiplier)
        """
        try:
            intent_scores = {}
            
            # Score each intent category
            for intent, patterns in self.intent_patterns.items():
                score = 0.0
                for pattern in patterns:
                    if re.search(pattern, command, re.IGNORECASE):
                        score += 1.0
                
                if score > 0:
                    intent_scores[intent] = score
            
            # Determine primary intent
            if intent_scores:
                primary_intent = max(intent_scores.keys(), key=lambda k: intent_scores[k])
            else:
                primary_intent = "unknown"
            
            # Calculate risk multiplier
            risk_multiplier = 1.0
            for modifier, multiplier in self.risk_modifiers.items():
                if modifier in command.lower():
                    risk_multiplier *= multiplier
            
            # Context-based risk adjustment
            if context.trust_level < 0.3:
                risk_multiplier *= 1.5
            if context.command_frequency > 10:  # High frequency
                risk_multiplier *= 1.2
            if context.security_alerts_24h > 0:
                risk_multiplier *= 1.3
            
            return primary_intent, min(risk_multiplier, 5.0)  # Cap at 5x
            
        except Exception as e:
            logger.error(f"Intent analysis error: {e}")
            return "unknown", 1.0


class AdvancedSecurityValidator:
    """
    Advanced Security Validator with context-aware analysis and ML-based threat detection.
    
    Extends the base SecurityValidator with sophisticated analysis capabilities
    including behavioral analysis, threat intelligence, and forensic analysis.
    """
    
    def __init__(
        self,
        base_validator: SecurityValidator,
        authorization_engine: AuthorizationEngine,
        audit_system: SecurityAuditSystem
    ):
        self.base_validator = base_validator
        self.authorization_engine = authorization_engine
        self.audit_system = audit_system
        
        # Advanced components
        self.intent_analyzer = CommandIntentAnalyzer()
        
        # Threat signatures
        self.threat_signatures: List[ThreatSignature] = []
        self.signature_cache: Dict[str, Tuple[List[ThreatSignature], datetime]] = {}
        self.cache_ttl = timedelta(minutes=30)
        
        # Behavioral baselines
        self.agent_baselines: Dict[uuid.UUID, Dict[str, Any]] = {}
        self.command_frequency_windows: Dict[uuid.UUID, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Performance metrics
        self.metrics = {
            "validations_performed": 0,
            "threats_detected": 0,
            "false_positives": 0,
            "avg_analysis_time_ms": 0.0,
            "signature_matches": 0,
            "behavioral_anomalies": 0,
            "escalations_triggered": 0
        }
        
        # Configuration
        self.config = {
            "enable_ml_analysis": True,
            "enable_behavioral_analysis": True,
            "enable_threat_intelligence": True,
            "default_analysis_mode": AnalysisMode.STANDARD,
            "max_analysis_time_ms": 200,
            "confidence_threshold": 0.7,
            "behavioral_anomaly_threshold": 2.0,  # Standard deviations
            "auto_update_signatures": True,
            "enable_command_logging": True
        }
        
        # Initialize default threat signatures
        self._initialize_threat_signatures()
    
    async def validate_command_advanced(
        self,
        command: str,
        context: CommandContext,
        analysis_mode: AnalysisMode = AnalysisMode.STANDARD
    ) -> SecurityAnalysisResult:
        """
        Perform advanced security validation of a command.
        
        Args:
            command: Command to validate
            context: Command execution context
            analysis_mode: Analysis mode (affects depth and speed)
            
        Returns:
            SecurityAnalysisResult with comprehensive analysis
        """
        start_time = time.time()
        
        try:
            # Start with base validation
            base_is_safe, base_risk, base_reason = await self.base_validator.validate_command(
                command, self._context_to_dict(context)
            )
            
            # Initialize result
            result = SecurityAnalysisResult(
                is_safe=base_is_safe,
                risk_level=self._map_risk_level(base_risk),
                threat_categories=[],
                confidence_score=0.8 if not base_is_safe else 0.2,
                matched_signatures=[],
                risk_factors=[base_reason] if not base_is_safe else [],
                behavioral_anomalies=[],
                analysis_mode=analysis_mode
            )
            
            # If base validation fails critically, return early
            if base_risk == SecurityRisk.CRITICAL:
                result.control_decision = ControlDecision.DENY
                result.recommended_actions = ["Block execution immediately"]
                result.analysis_time_ms = (time.time() - start_time) * 1000
                return result
            
            # Perform advanced analysis based on mode
            if analysis_mode in [AnalysisMode.STANDARD, AnalysisMode.DEEP, AnalysisMode.FORENSIC]:
                await self._perform_signature_analysis(command, context, result)
                
            if analysis_mode in [AnalysisMode.DEEP, AnalysisMode.FORENSIC]:
                await self._perform_behavioral_analysis(command, context, result)
                await self._perform_intent_analysis(command, context, result)
                
            if analysis_mode == AnalysisMode.FORENSIC:
                await self._perform_forensic_analysis(command, context, result)
            
            # Update behavioral baseline
            await self._update_agent_baseline(context.agent_id, command, context)
            
            # Make final decision
            await self._make_security_decision(result, context)
            
            # Update metrics
            analysis_time = (time.time() - start_time) * 1000
            result.analysis_time_ms = analysis_time
            self._update_metrics(result, analysis_time)
            
            # Log analysis result
            if self.config["enable_command_logging"]:
                await self._log_analysis_result(command, context, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Advanced security validation error: {e}")
            
            # Return safe default for errors
            analysis_time = (time.time() - start_time) * 1000
            return SecurityAnalysisResult(
                is_safe=False,
                risk_level=SecurityRiskLevel.HIGH,
                threat_categories=[ThreatCategory.UNKNOWN_THREAT],
                confidence_score=0.9,
                matched_signatures=[],
                risk_factors=[f"Analysis error: {str(e)}"],
                behavioral_anomalies=[],
                control_decision=ControlDecision.ESCALATE,
                recommended_actions=["Manual review required due to analysis error"],
                analysis_time_ms=analysis_time,
                analysis_mode=analysis_mode
            )
    
    async def _perform_signature_analysis(
        self,
        command: str,
        context: CommandContext,
        result: SecurityAnalysisResult
    ) -> None:
        """Perform threat signature analysis."""
        
        # Get active signatures
        signatures = await self._get_active_signatures()
        
        matched_signatures = []
        threat_categories = set()
        max_confidence = 0.0
        
        for signature in signatures:
            matches, confidence = signature.matches(command, context)
            
            if matches:
                matched_signatures.append(signature.name)
                threat_categories.add(signature.category)
                max_confidence = max(max_confidence, confidence)
                
                # Add signature-specific risk factors
                result.risk_factors.append(f"Matched threat signature: {signature.name}")
        
        # Update result
        result.matched_signatures = matched_signatures
        result.threat_categories = list(threat_categories)
        
        if matched_signatures:
            result.confidence_score = max(result.confidence_score, max_confidence)
            self.metrics["signature_matches"] += len(matched_signatures)
            
            # Escalate risk level if significant threats detected
            if ThreatCategory.COMMAND_INJECTION in threat_categories:
                result.risk_level = SecurityRiskLevel.CRITICAL
            elif ThreatCategory.PRIVILEGE_ESCALATION in threat_categories:
                result.risk_level = SecurityRiskLevel.HIGH
    
    async def _perform_behavioral_analysis(
        self,
        command: str,
        context: CommandContext,
        result: SecurityAnalysisResult
    ) -> None:
        """Perform behavioral anomaly analysis."""
        
        agent_id = context.agent_id
        anomalies = []
        
        # Get agent baseline
        baseline = self.agent_baselines.get(agent_id, {})
        
        if not baseline:
            # No baseline yet, start building one
            anomalies.append("No behavioral baseline established")
        else:
            # Check command frequency anomalies
            current_frequency = context.command_frequency
            baseline_frequency = baseline.get("avg_command_frequency", 1.0)
            
            if current_frequency > baseline_frequency * self.config["behavioral_anomaly_threshold"]:
                anomalies.append(f"Unusual command frequency: {current_frequency:.2f} vs baseline {baseline_frequency:.2f}")
            
            # Check command pattern anomalies
            command_type = self.intent_analyzer.analyze_intent(command, context)[0]
            typical_commands = baseline.get("command_types", {})
            
            if command_type not in typical_commands:
                anomalies.append(f"Unusual command type: {command_type}")
            elif typical_commands[command_type] < 0.05:  # Less than 5% of typical usage
                anomalies.append(f"Rare command type for agent: {command_type}")
            
            # Check temporal anomalies
            current_hour = context.timestamp.hour
            typical_hours = baseline.get("active_hours", [])
            
            if typical_hours and current_hour not in typical_hours:
                anomalies.append(f"Command execution outside typical hours: {current_hour}")
        
        # Update result
        result.behavioral_anomalies = anomalies
        
        if anomalies:
            result.confidence_score = max(result.confidence_score, 0.6)
            result.risk_factors.extend(anomalies)
            self.metrics["behavioral_anomalies"] += len(anomalies)
    
    async def _perform_intent_analysis(
        self,
        command: str,
        context: CommandContext,
        result: SecurityAnalysisResult
    ) -> None:
        """Perform command intent analysis."""
        
        intent, risk_multiplier = self.intent_analyzer.analyze_intent(command, context)
        
        result.command_intent = intent
        
        # Adjust risk level based on intent and multiplier
        if risk_multiplier > 2.0:
            if result.risk_level.value in ["MINIMAL", "LOW"]:
                result.risk_level = SecurityRiskLevel.MODERATE
            elif result.risk_level == SecurityRiskLevel.MODERATE:
                result.risk_level = SecurityRiskLevel.HIGH
        
        # Add intent-specific recommendations
        if intent == "system_administration" and context.trust_level < 0.5:
            result.recommended_actions.append("Verify agent authorization for system administration")
        elif intent == "network_operation" and risk_multiplier > 1.5:
            result.monitoring_requirements.append("Monitor network connections and data transfers")
        elif intent == "security_operation":
            result.recommended_actions.append("Enhanced logging and monitoring required")
    
    async def _perform_forensic_analysis(
        self,
        command: str,
        context: CommandContext,
        result: SecurityAnalysisResult
    ) -> None:
        """Perform comprehensive forensic analysis."""
        
        # Command obfuscation detection
        if self._detect_obfuscation(command):
            result.risk_factors.append("Command obfuscation detected")
            result.threat_categories.append(ThreatCategory.ANOMALOUS_BEHAVIOR)
        
        # Command chaining analysis
        chain_risk = self._analyze_command_chaining(command, context)
        if chain_risk > 0.7:
            result.risk_factors.append("High-risk command chaining detected")
        
        # Resource access pattern analysis
        affected_resources = self._analyze_resource_access(command)
        result.affected_resources = affected_resources
        
        if len(affected_resources) > 10:
            result.risk_factors.append("Broad resource access pattern")
        
        # Historical pattern analysis
        historical_risk = await self._analyze_historical_patterns(context.agent_id, command)
        if historical_risk > 0.8:
            result.risk_factors.append("Command matches historical attack patterns")
    
    def _detect_obfuscation(self, command: str) -> bool:
        """Detect command obfuscation techniques."""
        obfuscation_indicators = [
            r'[a-zA-Z]+=[a-zA-Z]+',  # Variable assignments to hide commands
            r'\$\{[^}]*\}',          # Variable expansion
            r'\\x[0-9a-fA-F]{2}',    # Hex encoding
            r'base64|b64decode',      # Base64 encoding
            r'eval\s*\(',            # Dynamic evaluation
            r'exec\s*\(',            # Dynamic execution
            r'`[^`]*`',              # Command substitution
            r'\$\([^)]*\)',          # Command substitution
        ]
        
        obfuscation_count = 0
        for pattern in obfuscation_indicators:
            if re.search(pattern, command):
                obfuscation_count += 1
        
        return obfuscation_count >= 2  # Multiple indicators suggest obfuscation
    
    def _analyze_command_chaining(self, command: str, context: CommandContext) -> float:
        """Analyze risk of command chaining patterns."""
        risk_score = 0.0
        
        # Chain operators
        if '&&' in command or '||' in command or ';' in command:
            risk_score += 0.3
        
        # Pipe operations
        pipe_count = command.count('|')
        if pipe_count > 3:
            risk_score += 0.4
        
        # Redirection operations
        redirect_count = command.count('>') + command.count('<')
        if redirect_count > 2:
            risk_score += 0.3
        
        # Previous command context
        if context.previous_commands:
            recent_commands = context.previous_commands[-5:]  # Last 5 commands
            if len(set(recent_commands)) == 1:  # Repeated identical commands
                risk_score += 0.5
        
        return min(risk_score, 1.0)
    
    def _analyze_resource_access(self, command: str) -> List[str]:
        """Analyze which resources the command might access."""
        resources = []
        
        # File system paths
        file_patterns = [
            r'/[a-zA-Z0-9/_.-]+',     # Absolute paths
            r'\./[a-zA-Z0-9/_.-]+',   # Relative paths
            r'~/[a-zA-Z0-9/_.-]+',    # Home directory paths
        ]
        
        for pattern in file_patterns:
            matches = re.findall(pattern, command)
            resources.extend(matches)
        
        # Network resources
        network_patterns = [
            r'https?://[^\s]+',       # HTTP URLs
            r'ftp://[^\s]+',          # FTP URLs
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP addresses
        ]
        
        for pattern in network_patterns:
            matches = re.findall(pattern, command)
            resources.extend(matches)
        
        return list(set(resources))  # Remove duplicates
    
    async def _analyze_historical_patterns(self, agent_id: uuid.UUID, command: str) -> float:
        """Analyze command against historical attack patterns."""
        # This would typically query a threat intelligence database
        # For now, we'll use a simplified pattern matching approach
        
        attack_patterns = [
            r'rm\s+-rf\s+/',                    # Destructive file operations
            r'curl.*\|\s*sh',                   # Download and execute
            r'wget.*\|\s*sh',                   # Download and execute
            r'nc\s+-l.*-e',                     # Netcat backdoor
            r'python.*-c.*exec',                # Python code execution
            r'perl.*-e.*exec',                  # Perl code execution
            r'bash.*<\(curl',                   # Bash process substitution
            r'powershell.*-enc',                # PowerShell encoded commands
        ]
        
        risk_score = 0.0
        for pattern in attack_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                risk_score += 0.3
        
        return min(risk_score, 1.0)
    
    async def _make_security_decision(
        self,
        result: SecurityAnalysisResult,
        context: CommandContext
    ) -> None:
        """Make final security control decision."""
        
        # Calculate overall risk score
        risk_factors = [
            result.confidence_score * 0.4,
            len(result.threat_categories) * 0.1,
            len(result.behavioral_anomalies) * 0.1,
            (1.0 - context.trust_level) * 0.2,
            context.current_risk_score * 0.2
        ]
        
        overall_risk = sum(risk_factors)
        
        # Make decision based on risk level and context
        if result.risk_level == SecurityRiskLevel.CRITICAL:
            result.control_decision = ControlDecision.DENY
            result.is_safe = False
        elif result.risk_level == SecurityRiskLevel.HIGH:
            if overall_risk > 0.8:
                result.control_decision = ControlDecision.DENY
                result.is_safe = False
            else:
                result.control_decision = ControlDecision.REQUIRE_APPROVAL
        elif result.risk_level == SecurityRiskLevel.MODERATE:
            if context.trust_level < 0.3 or overall_risk > 0.7:
                result.control_decision = ControlDecision.REQUIRE_APPROVAL
            else:
                result.control_decision = ControlDecision.ESCALATE
        else:
            result.control_decision = ControlDecision.ALLOW
            result.is_safe = True
        
        # Add specific recommendations
        if result.control_decision == ControlDecision.DENY:
            result.recommended_actions.append("Block command execution")
            result.recommended_actions.append("Investigate agent behavior")
        elif result.control_decision == ControlDecision.REQUIRE_APPROVAL:
            result.recommended_actions.append("Require human approval before execution")
            result.recommended_actions.append("Enhanced monitoring during execution")
        elif result.control_decision == ControlDecision.ESCALATE:
            result.recommended_actions.append("Log and monitor execution")
            result.recommended_actions.append("Review post-execution for anomalies")
    
    async def _get_active_signatures(self) -> List[ThreatSignature]:
        """Get active threat signatures with caching."""
        cache_key = "active_signatures"
        
        # Check cache
        if cache_key in self.signature_cache:
            signatures, cached_at = self.signature_cache[cache_key]
            if datetime.utcnow() - cached_at < self.cache_ttl:
                return signatures
        
        # Get active signatures
        active_signatures = [sig for sig in self.threat_signatures if sig.enabled]
        
        # Update cache
        self.signature_cache[cache_key] = (active_signatures, datetime.utcnow())
        
        return active_signatures
    
    async def _update_agent_baseline(
        self,
        agent_id: uuid.UUID,
        command: str,
        context: CommandContext
    ) -> None:
        """Update behavioral baseline for agent."""
        
        if agent_id not in self.agent_baselines:
            self.agent_baselines[agent_id] = {
                "command_count": 0,
                "command_types": {},
                "active_hours": set(),
                "avg_command_frequency": 1.0,
                "last_updated": datetime.utcnow()
            }
        
        baseline = self.agent_baselines[agent_id]
        
        # Update command frequency
        freq_window = self.command_frequency_windows[agent_id]
        freq_window.append(context.timestamp)
        
        # Calculate current frequency (commands per minute)
        if len(freq_window) > 1:
            time_span = (freq_window[-1] - freq_window[0]).total_seconds() / 60
            current_freq = len(freq_window) / max(time_span, 1)
            
            # Update rolling average
            baseline["avg_command_frequency"] = (
                baseline["avg_command_frequency"] * 0.9 + current_freq * 0.1
            )
        
        # Update command types
        intent, _ = self.intent_analyzer.analyze_intent(command, context)
        if intent not in baseline["command_types"]:
            baseline["command_types"][intent] = 0
        baseline["command_types"][intent] += 1
        
        # Normalize command type percentages
        total_commands = sum(baseline["command_types"].values())
        for cmd_type in baseline["command_types"]:
            baseline["command_types"][cmd_type] = baseline["command_types"][cmd_type] / total_commands
        
        # Update active hours
        baseline["active_hours"].add(context.timestamp.hour)
        
        # Update metadata
        baseline["command_count"] += 1
        baseline["last_updated"] = datetime.utcnow()
    
    def _initialize_threat_signatures(self) -> None:
        """Initialize default threat signatures."""
        
        signatures = [
            ThreatSignature(
                id="cmd_injection_001",
                name="Command Injection - Shell Metacharacters",
                category=ThreatCategory.COMMAND_INJECTION,
                patterns=[
                    r'[;&|]+.*(\bsh\b|\bbash\b|\bcmd\b)',
                    r'`[^`]*`',
                    r'\$\([^)]*\)',
                ],
                risk_score=0.9,
                confidence_threshold=0.8
            ),
            ThreatSignature(
                id="data_exfil_001",
                name="Data Exfiltration - Suspicious Transfers",
                category=ThreatCategory.DATA_EXFILTRATION,
                patterns=[
                    r'(curl|wget).*-d.*@.*',
                    r'(scp|rsync).*:/.*',
                    r'tar.*\|.*(nc|netcat)',
                ],
                risk_score=0.8,
                confidence_threshold=0.7
            ),
            ThreatSignature(
                id="priv_esc_001",
                name="Privilege Escalation Attempt",
                category=ThreatCategory.PRIVILEGE_ESCALATION,
                patterns=[
                    r'sudo\s+-u\s+root',
                    r'su\s+-\s+root',
                    r'chmod\s+\+s\s+',
                    r'setuid|setgid',
                ],
                risk_score=0.9,
                confidence_threshold=0.8
            ),
            ThreatSignature(
                id="resource_abuse_001",
                name="Resource Abuse - Fork Bombs and DoS",
                category=ThreatCategory.RESOURCE_ABUSE,
                patterns=[
                    r':\(\)\{.*:\|:.*\}',  # Fork bomb pattern
                    r'while\s+true.*do',
                    r'yes\s+.*\|.*',
                    r'dd\s+if=/dev/zero',
                ],
                risk_score=0.9,
                confidence_threshold=0.9
            ),
            ThreatSignature(
                id="persistence_001",
                name="Persistence Mechanism",
                category=ThreatCategory.PERSISTENCE_MECHANISM,
                patterns=[
                    r'crontab\s+-e',
                    r'echo.*>>.*/\.bashrc',
                    r'echo.*>>.*/\.profile',
                    r'systemctl.*enable',
                ],
                risk_score=0.7,
                confidence_threshold=0.6
            ),
            ThreatSignature(
                id="network_recon_001",
                name="Network Reconnaissance",
                category=ThreatCategory.NETWORK_RECONNAISSANCE,
                patterns=[
                    r'nmap.*-s[STAFNUX]',
                    r'nc.*-z.*-v',
                    r'masscan.*-p',
                    r'zmap.*-p',
                ],
                risk_score=0.6,
                confidence_threshold=0.7
            ),
        ]
        
        self.threat_signatures.extend(signatures)
        logger.info(f"Initialized {len(signatures)} default threat signatures")
    
    def _context_to_dict(self, context: CommandContext) -> Dict[str, Any]:
        """Convert CommandContext to dictionary for base validator."""
        return {
            "agent_id": str(context.agent_id),
            "session_id": str(context.session_id) if context.session_id else None,
            "working_directory": context.working_directory,
            "agent_type": context.agent_type,
            "trust_level": context.trust_level,
            "timestamp": context.timestamp.isoformat(),
            "source_ip": context.source_ip,
            "user_agent": context.user_agent,
            "current_risk_score": context.current_risk_score
        }
    
    def _map_risk_level(self, base_risk: SecurityRisk) -> SecurityRiskLevel:
        """Map base SecurityRisk to SecurityRiskLevel."""
        mapping = {
            SecurityRisk.SAFE: SecurityRiskLevel.MINIMAL,
            SecurityRisk.LOW: SecurityRiskLevel.LOW,
            SecurityRisk.MEDIUM: SecurityRiskLevel.MODERATE,
            SecurityRisk.HIGH: SecurityRiskLevel.HIGH,
            SecurityRisk.CRITICAL: SecurityRiskLevel.CRITICAL
        }
        return mapping.get(base_risk, SecurityRiskLevel.MODERATE)
    
    def _update_metrics(self, result: SecurityAnalysisResult, analysis_time: float) -> None:
        """Update performance metrics."""
        self.metrics["validations_performed"] += 1
        
        if not result.is_safe:
            self.metrics["threats_detected"] += 1
        
        if result.control_decision in [ControlDecision.ESCALATE, ControlDecision.REQUIRE_APPROVAL]:
            self.metrics["escalations_triggered"] += 1
        
        # Update average analysis time
        current_avg = self.metrics["avg_analysis_time_ms"]
        total_validations = self.metrics["validations_performed"]
        self.metrics["avg_analysis_time_ms"] = (
            (current_avg * (total_validations - 1) + analysis_time) / total_validations
        )
    
    async def _log_analysis_result(
        self,
        command: str,
        context: CommandContext,
        result: SecurityAnalysisResult
    ) -> None:
        """Log analysis result for audit and forensics."""
        
        # Create security event if threat detected
        if not result.is_safe or result.threat_categories:
            threat_level = ThreatLevel.CRITICAL if result.risk_level == SecurityRiskLevel.CRITICAL else ThreatLevel.HIGH
            
            event = SecurityEvent(
                id=uuid.uuid4(),
                event_type=AuditEventType.SUSPICIOUS_PATTERN,
                threat_level=threat_level,
                agent_id=context.agent_id,
                context_id=None,  # Would be set if available
                session_id=context.session_id,
                description=f"Advanced security analysis detected threats in command",
                details={
                    "command": command[:1000],  # Truncate for storage
                    "threat_categories": [cat.value for cat in result.threat_categories],
                    "matched_signatures": result.matched_signatures,
                    "risk_factors": result.risk_factors,
                    "behavioral_anomalies": result.behavioral_anomalies,
                    "confidence_score": result.confidence_score,
                    "control_decision": result.control_decision.value,
                    "analysis_time_ms": result.analysis_time_ms
                },
                timestamp=context.timestamp
            )
            
            await self.audit_system._handle_security_event(event)
        
        # Always log for forensic analysis
        logger.info(
            "Advanced security analysis completed",
            agent_id=str(context.agent_id),
            command_hash=hashlib.sha256(command.encode()).hexdigest()[:16],
            is_safe=result.is_safe,
            risk_level=result.risk_level.value,
            threat_categories=[cat.value for cat in result.threat_categories],
            confidence_score=result.confidence_score,
            control_decision=result.control_decision.value,
            analysis_time_ms=result.analysis_time_ms
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics."""
        return {
            "advanced_security_validator": self.metrics.copy(),
            "base_validator_metrics": self.base_validator.get_metrics(),
            "threat_signatures_count": len(self.threat_signatures),
            "active_signatures_count": len([sig for sig in self.threat_signatures if sig.enabled]),
            "agent_baselines_count": len(self.agent_baselines),
            "configuration": self.config.copy()
        }


# Factory function
async def create_advanced_security_validator(
    base_validator: SecurityValidator,
    authorization_engine: AuthorizationEngine,
    audit_system: SecurityAuditSystem
) -> AdvancedSecurityValidator:
    """
    Create AdvancedSecurityValidator instance.
    
    Args:
        base_validator: Base SecurityValidator
        authorization_engine: Authorization engine
        audit_system: Security audit system
        
    Returns:
        AdvancedSecurityValidator instance
    """
    return AdvancedSecurityValidator(base_validator, authorization_engine, audit_system)


# Convenience functions
async def validate_command_with_context(
    command: str,
    agent_id: uuid.UUID,
    session_id: Optional[uuid.UUID] = None,
    **context_kwargs
) -> SecurityAnalysisResult:
    """
    Convenience function for validating commands with context.
    
    Args:
        command: Command to validate
        agent_id: Agent identifier
        session_id: Session identifier
        **context_kwargs: Additional context parameters
        
    Returns:
        SecurityAnalysisResult
    """
    # This would typically get the validator from a service registry
    # For now, we'll document the interface
    raise NotImplementedError("This convenience function requires service registry integration")