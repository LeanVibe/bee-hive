"""
Integrated Security System for LeanVibe Agent Hive 2.0.

This system provides comprehensive integration of all security components into a unified
security framework that works seamlessly with the existing Hook Lifecycle System.

Features:
- Unified security validation pipeline
- Integration with existing security middleware and authorization engine
- Orchestrated threat detection and response
- Centralized security policy management
- Real-time security monitoring and alerting
- Comprehensive audit and forensic capabilities
- Performance-optimized security processing
"""

import asyncio
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import logging

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from .hook_lifecycle_system import SecurityValidator, HookLifecycleSystem, HookProcessingResult
from .advanced_security_validator import (
    AdvancedSecurityValidator, CommandContext, SecurityAnalysisResult, AnalysisMode
)
from .threat_detection_engine import ThreatDetectionEngine, ThreatDetection
from .security_policy_engine import SecurityPolicyEngine, PolicyEvaluationResult
from .enhanced_security_audit import EnhancedSecurityAudit
from .enhanced_security_safeguards import (
    EnhancedSecuritySafeguards, ControlDecision, SecurityContext, AgentBehaviorState
)
from .security_middleware import SecurityMiddleware
from .authorization_engine import AuthorizationEngine, AuthorizationResult
from .security_audit import SecurityAuditSystem
from ..models.agent import Agent

logger = structlog.get_logger()


class SecurityProcessingMode(Enum):
    """Security processing modes for different scenarios."""
    FAST = "FAST"           # <10ms, basic validation only
    STANDARD = "STANDARD"   # <50ms, full pipeline
    DEEP = "DEEP"          # <200ms, advanced analysis
    FORENSIC = "FORENSIC"  # No time limit, comprehensive analysis


@dataclass
class SecurityProcessingContext:
    """Comprehensive context for security processing."""
    agent_id: uuid.UUID
    session_id: Optional[uuid.UUID]
    command: str
    
    # Processing configuration
    processing_mode: SecurityProcessingMode = SecurityProcessingMode.STANDARD
    skip_policy_evaluation: bool = False
    skip_threat_detection: bool = False
    skip_behavioral_analysis: bool = False
    
    # Context data
    working_directory: Optional[str] = None
    environment_vars: Dict[str, str] = field(default_factory=dict)
    previous_commands: List[str] = field(default_factory=list)
    user_context: Optional[str] = None
    
    # Agent context
    agent_type: Optional[str] = None
    agent_capabilities: List[str] = field(default_factory=list)
    trust_level: float = 0.5
    
    # Network context
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    geo_location: Optional[Dict[str, str]] = None
    
    # Timing context
    timestamp: datetime = field(default_factory=datetime.utcnow)
    time_since_last_command: Optional[timedelta] = None
    command_frequency: float = 0.0
    
    # Risk context
    current_risk_score: float = 0.0
    recent_anomalies: List[str] = field(default_factory=list)
    security_alerts_24h: int = 0
    
    def to_command_context(self) -> CommandContext:
        """Convert to CommandContext for AdvancedSecurityValidator."""
        return CommandContext(
            agent_id=self.agent_id,
            session_id=self.session_id,
            command=self.command,
            working_directory=self.working_directory,
            environment_vars=self.environment_vars,
            previous_commands=self.previous_commands,
            user_context=self.user_context,
            agent_type=self.agent_type,
            agent_capabilities=self.agent_capabilities,
            trust_level=self.trust_level,
            timestamp=self.timestamp,
            time_since_last_command=self.time_since_last_command,
            command_frequency=self.command_frequency,
            source_ip=self.source_ip,
            user_agent=self.user_agent,
            geo_location=self.geo_location,
            current_risk_score=self.current_risk_score,
            recent_anomalies=self.recent_anomalies,
            security_alerts_24h=self.security_alerts_24h
        )


@dataclass
class IntegratedSecurityResult:
    """Comprehensive result from integrated security processing."""
    # Final decision
    control_decision: ControlDecision
    is_safe: bool
    overall_confidence: float
    
    # Component results
    base_validation_result: Tuple[bool, Any, str]  # (is_safe, risk_level, reason)
    advanced_analysis_result: SecurityAnalysisResult
    threat_detections: List[ThreatDetection]
    policy_evaluation_result: PolicyEvaluationResult
    
    # Processing metadata
    processing_mode: SecurityProcessingMode
    total_processing_time_ms: float
    components_used: List[str]
    
    # Recommendations and actions
    recommended_actions: List[str] = field(default_factory=list)
    monitoring_requirements: List[str] = field(default_factory=list)
    escalation_required: bool = False
    audit_required: bool = False
    
    # Performance breakdown
    component_timings: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "control_decision": self.control_decision.value,
            "is_safe": self.is_safe,
            "overall_confidence": self.overall_confidence,
            "advanced_analysis": self.advanced_analysis_result.__dict__ if self.advanced_analysis_result else None,
            "threat_detections": [td.to_dict() for td in self.threat_detections],
            "policy_evaluation": self.policy_evaluation_result.to_dict() if self.policy_evaluation_result else None,
            "processing_mode": self.processing_mode.value,
            "total_processing_time_ms": self.total_processing_time_ms,
            "components_used": self.components_used,
            "recommended_actions": self.recommended_actions,
            "monitoring_requirements": self.monitoring_requirements,
            "escalation_required": self.escalation_required,
            "audit_required": self.audit_required,
            "component_timings": self.component_timings
        }


class IntegratedSecuritySystem:
    """
    Integrated Security System that orchestrates all security components.
    
    Provides a unified interface for comprehensive security processing that
    integrates with the existing Hook Lifecycle System and security infrastructure.
    """
    
    def __init__(
        self,
        hook_lifecycle_system: HookLifecycleSystem,
        advanced_validator: AdvancedSecurityValidator,
        threat_detection_engine: ThreatDetectionEngine,
        policy_engine: SecurityPolicyEngine,
        audit_system: EnhancedSecurityAudit,
        authorization_engine: AuthorizationEngine,
        enhanced_safeguards: EnhancedSecuritySafeguards
    ):
        self.hook_system = hook_lifecycle_system
        self.advanced_validator = advanced_validator
        self.threat_engine = threat_detection_engine
        self.policy_engine = policy_engine
        self.audit_system = audit_system
        self.authorization_engine = authorization_engine
        self.enhanced_safeguards = enhanced_safeguards
        
        # Get the base security validator from hook system
        self.base_validator = self.hook_system.security_validator
        
        # Performance metrics
        self.metrics = {
            "requests_processed": 0,
            "threats_detected": 0,
            "policies_triggered": 0,
            "commands_blocked": 0,
            "approvals_required": 0,
            "avg_processing_time_ms": 0.0,
            "fast_mode_used": 0,
            "standard_mode_used": 0,
            "deep_mode_used": 0,
            "forensic_mode_used": 0,
            "component_usage": {
                "base_validator": 0,
                "advanced_validator": 0,
                "threat_detection": 0,
                "policy_engine": 0,
                "audit_system": 0
            }
        }
        
        # Configuration
        self.config = {
            "enable_advanced_validation": True,
            "enable_threat_detection": True,
            "enable_policy_evaluation": True,
            "enable_behavioral_analysis": True,
            "fast_mode_threshold_ms": 10,
            "standard_mode_threshold_ms": 50,
            "deep_mode_threshold_ms": 200,
            "auto_mode_selection": True,
            "parallel_processing": True,
            "max_processing_time_ms": 500
        }
        
        # Processing pipeline configuration
        self.pipeline_config = {
            SecurityProcessingMode.FAST: {
                "use_base_validator": True,
                "use_advanced_validator": False,
                "use_threat_detection": False,
                "use_policy_engine": False,
                "analysis_mode": None
            },
            SecurityProcessingMode.STANDARD: {
                "use_base_validator": True,
                "use_advanced_validator": True,
                "use_threat_detection": True,
                "use_policy_engine": True,
                "analysis_mode": AnalysisMode.STANDARD
            },
            SecurityProcessingMode.DEEP: {
                "use_base_validator": True,
                "use_advanced_validator": True,
                "use_threat_detection": True,
                "use_policy_engine": True,
                "analysis_mode": AnalysisMode.DEEP
            },
            SecurityProcessingMode.FORENSIC: {
                "use_base_validator": True,
                "use_advanced_validator": True,
                "use_threat_detection": True,
                "use_policy_engine": True,
                "analysis_mode": AnalysisMode.FORENSIC
            }
        }
    
    async def process_security_validation(
        self,
        context: SecurityProcessingContext
    ) -> IntegratedSecurityResult:
        """
        Process comprehensive security validation through the integrated pipeline.
        
        Args:
            context: Security processing context
            
        Returns:
            IntegratedSecurityResult with comprehensive analysis
        """
        start_time = time.time()
        component_timings = {}
        components_used = []
        
        try:
            # Select processing mode if auto-selection is enabled
            if self.config["auto_mode_selection"] and context.processing_mode == SecurityProcessingMode.STANDARD:
                context.processing_mode = await self._select_processing_mode(context)
            
            # Update mode usage metrics
            mode_metric_key = f"{context.processing_mode.value.lower()}_mode_used"
            self.metrics[mode_metric_key] += 1
            
            # Get pipeline configuration
            pipeline_cfg = self.pipeline_config[context.processing_mode]
            
            # Initialize results
            base_validation_result = None
            advanced_analysis_result = None
            threat_detections = []
            policy_evaluation_result = None
            
            # Phase 1: Base Security Validation
            if pipeline_cfg["use_base_validator"]:
                base_start = time.time()
                base_validation_result = await self._run_base_validation(context)
                component_timings["base_validator"] = (time.time() - base_start) * 1000
                components_used.append("base_validator")
                self.metrics["component_usage"]["base_validator"] += 1
            
            # Phase 2: Advanced Security Analysis
            if pipeline_cfg["use_advanced_validator"] and not context.skip_behavioral_analysis:
                adv_start = time.time()
                command_context = context.to_command_context()
                
                advanced_analysis_result = await self.advanced_validator.validate_command_advanced(
                    context.command,
                    command_context,
                    pipeline_cfg["analysis_mode"]
                )
                
                component_timings["advanced_validator"] = (time.time() - adv_start) * 1000
                components_used.append("advanced_validator")
                self.metrics["component_usage"]["advanced_validator"] += 1
            
            # Phase 3: Threat Detection (can run in parallel with policy evaluation)
            if (pipeline_cfg["use_threat_detection"] and 
                not context.skip_threat_detection and 
                advanced_analysis_result):
                
                threat_start = time.time()
                threat_detections = await self.threat_engine.analyze_agent_behavior(
                    context.agent_id,
                    context.command,
                    context.to_command_context(),
                    advanced_analysis_result
                )
                component_timings["threat_detection"] = (time.time() - threat_start) * 1000
                components_used.append("threat_detection")
                self.metrics["component_usage"]["threat_detection"] += 1
            
            # Phase 4: Policy Evaluation
            if (pipeline_cfg["use_policy_engine"] and 
                not context.skip_policy_evaluation and 
                advanced_analysis_result):
                
                policy_start = time.time()
                policy_evaluation_result = await self.policy_engine.evaluate_policies(
                    context.to_command_context(),
                    advanced_analysis_result,
                    threat_detections
                )
                component_timings["policy_engine"] = (time.time() - policy_start) * 1000
                components_used.append("policy_engine")
                self.metrics["component_usage"]["policy_engine"] += 1
            
            # Phase 5: Decision Integration and Finalization
            integrated_result = await self._integrate_results(
                context,
                base_validation_result,
                advanced_analysis_result,
                threat_detections,
                policy_evaluation_result,
                component_timings,
                components_used
            )
            
            # Phase 6: Audit and Logging
            if integrated_result.audit_required or not integrated_result.is_safe:
                audit_start = time.time()
                await self.audit_system.log_security_analysis_result(
                    context.to_command_context(),
                    advanced_analysis_result or SecurityAnalysisResult(
                        is_safe=base_validation_result[0] if base_validation_result else True,
                        risk_level=self._map_base_risk_to_level(base_validation_result[1]) if base_validation_result else None,
                        threat_categories=[],
                        confidence_score=0.5,
                        matched_signatures=[],
                        risk_factors=[base_validation_result[2]] if base_validation_result else [],
                        behavioral_anomalies=[],
                        control_decision=integrated_result.control_decision,
                        recommended_actions=integrated_result.recommended_actions,
                        monitoring_requirements=integrated_result.monitoring_requirements,
                        analysis_time_ms=integrated_result.total_processing_time_ms,
                        analysis_mode=pipeline_cfg.get("analysis_mode", AnalysisMode.STANDARD)
                    ),
                    threat_detections,
                    policy_evaluation_result
                )
                component_timings["audit_system"] = (time.time() - audit_start) * 1000
                components_used.append("audit_system")
                self.metrics["component_usage"]["audit_system"] += 1
            
            # Update final timing
            integrated_result.total_processing_time_ms = (time.time() - start_time) * 1000
            integrated_result.component_timings = component_timings
            
            # Update metrics
            self._update_metrics(integrated_result)
            
            return integrated_result
            
        except Exception as e:
            logger.error(f"Integrated security processing error: {e}")
            
            # Return safe default on error
            processing_time = (time.time() - start_time) * 1000
            return IntegratedSecurityResult(
                control_decision=ControlDecision.ESCALATE,
                is_safe=False,
                overall_confidence=1.0,
                base_validation_result=(False, "HIGH", f"Processing error: {str(e)}"),
                advanced_analysis_result=None,
                threat_detections=[],
                policy_evaluation_result=None,
                processing_mode=context.processing_mode,
                total_processing_time_ms=processing_time,
                components_used=components_used,
                recommended_actions=["Manual review required due to processing error"],
                escalation_required=True,
                audit_required=True,
                component_timings=component_timings
            )
    
    async def validate_hook_with_integrated_security(
        self,
        agent_id: uuid.UUID,
        session_id: Optional[uuid.UUID],
        command: str,
        **context_kwargs
    ) -> HookProcessingResult:
        """
        Validate hook through integrated security system and return HookProcessingResult.
        
        This method provides compatibility with the existing Hook Lifecycle System
        while leveraging the full integrated security pipeline.
        
        Args:
            agent_id: Agent identifier
            session_id: Session identifier
            command: Command to validate
            **context_kwargs: Additional context parameters
            
        Returns:
            HookProcessingResult compatible with Hook Lifecycle System
        """
        try:
            # Build security processing context
            processing_context = SecurityProcessingContext(
                agent_id=agent_id,
                session_id=session_id,
                command=command,
                **context_kwargs
            )
            
            # Process security validation
            security_result = await self.process_security_validation(processing_context)
            
            # Convert to HookProcessingResult
            hook_result = HookProcessingResult(
                success=(security_result.control_decision != ControlDecision.DENY),
                processing_time_ms=security_result.total_processing_time_ms,
                security_decision=security_result.control_decision,
                blocked_reason=None if security_result.is_safe else "Security validation failed",
                event_id=str(uuid.uuid4())
            )
            
            # Add blocked reason if needed
            if not security_result.is_safe:
                reasons = []
                if security_result.advanced_analysis_result:
                    reasons.extend(security_result.advanced_analysis_result.risk_factors)
                if security_result.threat_detections:
                    reasons.extend([
                        f"Threat detected: {td.threat_type.value}" 
                        for td in security_result.threat_detections
                    ])
                hook_result.blocked_reason = "; ".join(reasons[:3])  # Top 3 reasons
            
            return hook_result
            
        except Exception as e:
            logger.error(f"Hook validation error: {e}")
            return HookProcessingResult(
                success=False,
                processing_time_ms=0.0,
                error=str(e),
                security_decision=ControlDecision.ESCALATE,
                blocked_reason="Security validation error"
            )
    
    async def _select_processing_mode(
        self,
        context: SecurityProcessingContext
    ) -> SecurityProcessingMode:
        """Automatically select appropriate processing mode based on context."""
        
        # High-risk indicators suggest deeper analysis
        risk_indicators = 0
        
        if context.current_risk_score > 0.7:
            risk_indicators += 2
        elif context.current_risk_score > 0.5:
            risk_indicators += 1
        
        if context.security_alerts_24h > 0:
            risk_indicators += 1
        
        if context.recent_anomalies:
            risk_indicators += len(context.recent_anomalies)
        
        if context.trust_level < 0.3:
            risk_indicators += 2
        elif context.trust_level < 0.5:
            risk_indicators += 1
        
        # Command complexity indicators
        if len(context.command) > 200:
            risk_indicators += 1
        
        if any(keyword in context.command.lower() for keyword in ['sudo', 'rm', 'chmod', 'wget', 'curl']):
            risk_indicators += 1
        
        # Select mode based on risk indicators
        if risk_indicators >= 4:
            return SecurityProcessingMode.DEEP
        elif risk_indicators >= 2:
            return SecurityProcessingMode.STANDARD
        else:
            return SecurityProcessingMode.FAST
    
    async def _run_base_validation(
        self,
        context: SecurityProcessingContext
    ) -> Tuple[bool, Any, str]:
        """Run base security validation."""
        
        # Convert context to format expected by base validator
        base_context = {
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
        
        return await self.base_validator.validate_command(context.command, base_context)
    
    async def _integrate_results(
        self,
        context: SecurityProcessingContext,
        base_result: Optional[Tuple[bool, Any, str]],
        advanced_result: Optional[SecurityAnalysisResult],
        threat_detections: List[ThreatDetection],
        policy_result: Optional[PolicyEvaluationResult],
        component_timings: Dict[str, float],
        components_used: List[str]
    ) -> IntegratedSecurityResult:
        """Integrate results from all security components."""
        
        # Initialize with safe defaults
        control_decision = ControlDecision.ALLOW
        is_safe = True
        overall_confidence = 0.0
        recommended_actions = []
        monitoring_requirements = []
        escalation_required = False
        audit_required = False
        
        # Process base validation result
        if base_result:
            base_is_safe, base_risk, base_reason = base_result
            if not base_is_safe:
                is_safe = False
                if base_risk == "CRITICAL":
                    control_decision = ControlDecision.DENY
                    recommended_actions.append("Critical security threat detected")
                elif base_risk == "HIGH":
                    control_decision = ControlDecision.REQUIRE_APPROVAL
                    recommended_actions.append("High-risk command requires approval")
                overall_confidence = max(overall_confidence, 0.8)
        
        # Process advanced analysis result
        if advanced_result:
            if not advanced_result.is_safe:
                is_safe = False
                # Advanced analysis takes precedence for control decisions
                control_decision = advanced_result.control_decision
                recommended_actions.extend(advanced_result.recommended_actions)
                monitoring_requirements.extend(advanced_result.monitoring_requirements)
            
            overall_confidence = max(overall_confidence, advanced_result.confidence_score)
            
            if advanced_result.behavioral_anomalies:
                audit_required = True
                monitoring_requirements.append("Behavioral anomaly monitoring")
        
        # Process threat detections
        if threat_detections:
            is_safe = False
            audit_required = True
            
            # Check for critical threats
            critical_threats = [td for td in threat_detections if td.severity.value == "CRITICAL"]
            high_threats = [td for td in threat_detections if td.severity.value == "HIGH"]
            
            if critical_threats:
                control_decision = ControlDecision.DENY
                escalation_required = True
                recommended_actions.append("Critical threats detected - immediate containment required")
            elif high_threats:
                if control_decision == ControlDecision.ALLOW:
                    control_decision = ControlDecision.REQUIRE_APPROVAL
                escalation_required = True
                recommended_actions.append("High-severity threats detected")
            
            # Add threat-specific recommendations
            for detection in threat_detections:
                recommended_actions.extend(detection.recommended_actions)
            
            # Update confidence based on threat detection confidence
            threat_confidence = max([td.confidence for td in threat_detections], default=0.0)
            overall_confidence = max(overall_confidence, threat_confidence)
        
        # Process policy evaluation result
        if policy_result:
            # Policy decisions can override other decisions
            if policy_result.decision != ControlDecision.ALLOW:
                control_decision = policy_result.decision
                is_safe = (control_decision == ControlDecision.ALLOW)
            
            if policy_result.escalation_required:
                escalation_required = True
            
            if policy_result.audit_required:
                audit_required = True
            
            recommended_actions.extend(policy_result.recommended_actions)
            monitoring_requirements.extend(policy_result.monitoring_requirements)
            
            # Update confidence based on policy confidence
            overall_confidence = max(overall_confidence, policy_result.confidence)
        
        # Final safety check - if we have any negative indicators, mark as unsafe
        if (threat_detections or 
            (advanced_result and not advanced_result.is_safe) or
            (base_result and not base_result[0])):
            is_safe = False
        
        # Ensure control decision matches safety assessment
        if not is_safe and control_decision == ControlDecision.ALLOW:
            control_decision = ControlDecision.ESCALATE
            escalation_required = True
        
        # Remove duplicate recommendations
        recommended_actions = list(set(recommended_actions))
        monitoring_requirements = list(set(monitoring_requirements))
        
        return IntegratedSecurityResult(
            control_decision=control_decision,
            is_safe=is_safe,
            overall_confidence=overall_confidence,
            base_validation_result=base_result,
            advanced_analysis_result=advanced_result,
            threat_detections=threat_detections,
            policy_evaluation_result=policy_result,
            processing_mode=context.processing_mode,
            total_processing_time_ms=sum(component_timings.values()),
            components_used=components_used,
            recommended_actions=recommended_actions,
            monitoring_requirements=monitoring_requirements,
            escalation_required=escalation_required,
            audit_required=audit_required,
            component_timings=component_timings
        )
    
    def _map_base_risk_to_level(self, base_risk: Any):
        """Map base risk level to SecurityRiskLevel."""
        from .enhanced_security_safeguards import SecurityRiskLevel
        
        if hasattr(base_risk, 'value'):
            risk_str = base_risk.value
        else:
            risk_str = str(base_risk).upper()
        
        mapping = {
            "SAFE": SecurityRiskLevel.MINIMAL,
            "LOW": SecurityRiskLevel.LOW,
            "MEDIUM": SecurityRiskLevel.MODERATE,
            "HIGH": SecurityRiskLevel.HIGH,
            "CRITICAL": SecurityRiskLevel.CRITICAL
        }
        
        return mapping.get(risk_str, SecurityRiskLevel.MODERATE)
    
    def _update_metrics(self, result: IntegratedSecurityResult) -> None:
        """Update system performance metrics."""
        
        self.metrics["requests_processed"] += 1
        
        if result.threat_detections:
            self.metrics["threats_detected"] += len(result.threat_detections)
        
        if result.policy_evaluation_result and result.policy_evaluation_result.matched_policies:
            self.metrics["policies_triggered"] += len(result.policy_evaluation_result.matched_policies)
        
        if result.control_decision == ControlDecision.DENY:
            self.metrics["commands_blocked"] += 1
        elif result.control_decision == ControlDecision.REQUIRE_APPROVAL:
            self.metrics["approvals_required"] += 1
        
        # Update average processing time
        current_avg = self.metrics["avg_processing_time_ms"]
        total_requests = self.metrics["requests_processed"]
        self.metrics["avg_processing_time_ms"] = (
            (current_avg * (total_requests - 1) + result.total_processing_time_ms) / total_requests
        )
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from all security components."""
        
        return {
            "integrated_security_system": self.metrics.copy(),
            "hook_lifecycle_system": self.hook_system.get_comprehensive_metrics(),
            "advanced_validator": self.advanced_validator.get_metrics(),
            "threat_detection_engine": self.threat_engine.get_metrics(),
            "policy_engine": self.policy_engine.get_metrics(),
            "audit_system": self.audit_system.get_metrics(),
            "enhanced_safeguards": self.enhanced_safeguards.get_comprehensive_metrics(),
            "configuration": self.config.copy(),
            "pipeline_configuration": self.pipeline_config
        }
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get overall security system status."""
        
        total_requests = self.metrics["requests_processed"]
        
        return {
            "system_health": "operational",
            "total_requests_processed": total_requests,
            "threats_detected": self.metrics["threats_detected"],
            "threat_detection_rate": self.metrics["threats_detected"] / max(total_requests, 1),
            "commands_blocked": self.metrics["commands_blocked"],
            "block_rate": self.metrics["commands_blocked"] / max(total_requests, 1),
            "approvals_required": self.metrics["approvals_required"],
            "approval_rate": self.metrics["approvals_required"] / max(total_requests, 1),
            "avg_processing_time_ms": self.metrics["avg_processing_time_ms"],
            "processing_mode_distribution": {
                "fast": self.metrics["fast_mode_used"],
                "standard": self.metrics["standard_mode_used"],
                "deep": self.metrics["deep_mode_used"],
                "forensic": self.metrics["forensic_mode_used"]
            },
            "component_usage": self.metrics["component_usage"].copy(),
            "system_configuration": {
                "advanced_validation_enabled": self.config["enable_advanced_validation"],
                "threat_detection_enabled": self.config["enable_threat_detection"],
                "policy_evaluation_enabled": self.config["enable_policy_evaluation"],
                "behavioral_analysis_enabled": self.config["enable_behavioral_analysis"],
                "auto_mode_selection": self.config["auto_mode_selection"],
                "parallel_processing": self.config["parallel_processing"]
            }
        }
    
    async def perform_security_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive security system health check."""
        
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "component_status": {},
            "performance_metrics": {},
            "recommendations": []
        }
        
        try:
            # Check component health
            components = {
                "hook_lifecycle_system": self.hook_system,
                "advanced_validator": self.advanced_validator,
                "threat_detection_engine": self.threat_engine,
                "policy_engine": self.policy_engine,
                "audit_system": self.audit_system,
                "authorization_engine": self.authorization_engine,
                "enhanced_safeguards": self.enhanced_safeguards
            }
            
            for name, component in components.items():
                try:
                    if hasattr(component, 'get_metrics'):
                        metrics = component.get_metrics()
                        health_status["component_status"][name] = "healthy"
                        health_status["performance_metrics"][name] = metrics
                    else:
                        health_status["component_status"][name] = "unknown"
                except Exception as e:
                    health_status["component_status"][name] = f"error: {str(e)}"
                    health_status["overall_status"] = "degraded"
            
            # Performance analysis
            avg_processing_time = self.metrics["avg_processing_time_ms"]
            if avg_processing_time > self.config["max_processing_time_ms"]:
                health_status["recommendations"].append(
                    f"Average processing time ({avg_processing_time:.2f}ms) exceeds target "
                    f"({self.config['max_processing_time_ms']}ms)"
                )
                health_status["overall_status"] = "warning"
            
            # Threat detection effectiveness
            total_requests = self.metrics["requests_processed"]
            if total_requests > 100:  # Only analyze if we have sufficient data
                threat_rate = self.metrics["threats_detected"] / total_requests
                if threat_rate > 0.1:  # More than 10% threat rate
                    health_status["recommendations"].append(
                        f"High threat detection rate ({threat_rate:.2%}) - review agent behavior patterns"
                    )
                elif threat_rate < 0.001:  # Less than 0.1% threat rate
                    health_status["recommendations"].append(
                        "Very low threat detection rate - consider adjusting sensitivity"
                    )
            
            # Component usage balance
            component_usage = self.metrics["component_usage"]
            total_usage = sum(component_usage.values())
            if total_usage > 0:
                for component, usage_count in component_usage.items():
                    usage_rate = usage_count / total_usage
                    if usage_rate < 0.1 and self.config.get(f"enable_{component.replace('_', '_')}", True):
                        health_status["recommendations"].append(
                            f"Low usage of {component} component - verify configuration"
                        )
            
            return health_status
            
        except Exception as e:
            logger.error(f"Security health check failed: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_status": "error",
                "error": str(e),
                "recommendations": ["Manual system inspection required"]
            }


# Factory function
async def create_integrated_security_system(
    hook_lifecycle_system: HookLifecycleSystem,
    advanced_validator: AdvancedSecurityValidator,
    threat_detection_engine: ThreatDetectionEngine,
    policy_engine: SecurityPolicyEngine,
    audit_system: EnhancedSecurityAudit,
    authorization_engine: AuthorizationEngine,
    enhanced_safeguards: EnhancedSecuritySafeguards
) -> IntegratedSecuritySystem:
    """
    Create IntegratedSecuritySystem instance.
    
    Args:
        hook_lifecycle_system: Hook lifecycle system
        advanced_validator: Advanced security validator
        threat_detection_engine: Threat detection engine
        policy_engine: Security policy engine
        audit_system: Enhanced security audit system
        authorization_engine: Authorization engine
        enhanced_safeguards: Enhanced security safeguards
        
    Returns:
        IntegratedSecuritySystem instance
    """
    return IntegratedSecuritySystem(
        hook_lifecycle_system,
        advanced_validator,
        threat_detection_engine,
        policy_engine,
        audit_system,
        authorization_engine,
        enhanced_safeguards
    )


# Convenience function for hook integration
async def integrate_with_hook_system(
    hook_system: HookLifecycleSystem,
    integrated_security: IntegratedSecuritySystem
) -> None:
    """
    Integrate the advanced security system with the hook lifecycle system.
    
    This function replaces the base security validator in the hook system
    with the integrated security system while maintaining compatibility.
    
    Args:
        hook_system: Hook lifecycle system to integrate with
        integrated_security: Integrated security system
    """
    
    # Create a wrapper that provides the interface expected by the hook system
    class IntegratedSecurityWrapper:
        def __init__(self, integrated_system: IntegratedSecuritySystem):
            self.integrated_system = integrated_system
            # Preserve original validator for fallback
            self.original_validator = hook_system.security_validator
        
        async def validate_command(
            self,
            command: str,
            context: Optional[Dict[str, Any]] = None
        ) -> Tuple[bool, Any, str]:
            """Validate command using integrated security system."""
            try:
                # Build processing context
                context = context or {}
                processing_context = SecurityProcessingContext(
                    agent_id=uuid.UUID(context.get("agent_id", str(uuid.uuid4()))),
                    session_id=uuid.UUID(context.get("session_id")) if context.get("session_id") else None,
                    command=command,
                    working_directory=context.get("working_directory"),
                    agent_type=context.get("agent_type"),
                    trust_level=context.get("trust_level", 0.5),
                    source_ip=context.get("source_ip"),
                    user_agent=context.get("user_agent"),
                    current_risk_score=context.get("current_risk_score", 0.0),
                    processing_mode=SecurityProcessingMode.FAST  # Use fast mode for hook validation
                )
                
                # Process through integrated system
                result = await self.integrated_system.process_security_validation(processing_context)
                
                # Convert to expected format
                risk_level = "SAFE"
                if not result.is_safe:
                    if result.control_decision == ControlDecision.DENY:
                        risk_level = "CRITICAL"
                    elif result.control_decision == ControlDecision.REQUIRE_APPROVAL:
                        risk_level = "HIGH"
                    else:
                        risk_level = "MEDIUM"
                
                reason = "; ".join(result.recommended_actions[:2]) if result.recommended_actions else "Security validation completed"
                
                return result.is_safe, risk_level, reason
                
            except Exception as e:
                logger.error(f"Integrated security validation error: {e}")
                # Fall back to original validator
                return await self.original_validator.validate_command(command, context)
        
        # Delegate other methods to original validator
        def __getattr__(self, name):
            return getattr(self.original_validator, name)
    
    # Replace the security validator
    hook_system.security_validator = IntegratedSecurityWrapper(integrated_security)
    
    logger.info("Integrated security system successfully integrated with hook lifecycle system")