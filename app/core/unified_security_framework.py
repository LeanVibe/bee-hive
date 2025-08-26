"""
Unified Security Framework for LeanVibe Agent Hive 2.0
======================================================

Production-ready security framework that integrates all security components
into a cohesive, enterprise-grade security system. This framework provides
centralized security orchestration, policy management, and compliance validation.

Epic 3 - Security & Operations: Core Security Integration
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Callable
import structlog

from .advanced_security_validator import AdvancedSecurityValidator, ThreatCategory
from .api_security_middleware import APISecurityMiddleware, RateLimitStrategy
from .enhanced_security_audit import EnhancedSecurityAudit, AuditEventType
from .enhanced_security_safeguards import EnhancedSecuritySafeguards, SecurityRiskLevel
from .enterprise_security_system import EnterpriseSecuritySystem, SecurityLevel
from .integrated_security_system import IntegratedSecuritySystem
from .security_monitoring_system import SecurityMonitoringSystem
from .security_policy_engine import SecurityPolicyEngine
from .redis import get_redis_client
from ..observability.intelligent_alerting_system import IntelligentAlertingSystem

logger = structlog.get_logger()


class SecurityFrameworkStatus(Enum):
    """Security framework operational status."""
    INITIALIZING = "initializing"
    OPERATIONAL = "operational"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"
    EMERGENCY = "emergency"


class SecurityPolicyType(Enum):
    """Types of security policies managed by the framework."""
    ACCESS_CONTROL = "access_control"
    API_SECURITY = "api_security"
    DATA_PROTECTION = "data_protection"
    THREAT_PREVENTION = "threat_prevention"
    COMPLIANCE = "compliance"
    AUDIT_LOGGING = "audit_logging"


@dataclass
class SecurityFrameworkConfig:
    """Configuration for the unified security framework."""
    enable_advanced_validation: bool = True
    enable_api_middleware: bool = True
    enable_security_audit: bool = True
    enable_threat_monitoring: bool = True
    enable_enterprise_controls: bool = True
    
    # Rate limiting configuration
    default_rate_limit_strategy: RateLimitStrategy = RateLimitStrategy.ADAPTIVE
    default_requests_per_minute: int = 100
    default_burst_size: int = 20
    
    # Security levels
    minimum_security_level: SecurityLevel = SecurityLevel.INTERNAL
    audit_retention_days: int = 365
    threat_detection_sensitivity: float = 0.8
    
    # Alerting configuration
    enable_real_time_alerts: bool = True
    alert_escalation_timeout: int = 300  # 5 minutes
    critical_incident_timeout: int = 60   # 1 minute


@dataclass
class SecurityMetrics:
    """Security framework metrics for monitoring and reporting."""
    total_requests_validated: int = 0
    blocked_requests: int = 0
    security_incidents: int = 0
    false_positives: int = 0
    average_validation_time_ms: float = 0.0
    threat_detection_accuracy: float = 0.0
    compliance_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SecurityReport:
    """Comprehensive security validation report."""
    request_id: str
    validation_result: bool
    risk_level: SecurityRiskLevel
    threat_categories: List[ThreatCategory]
    policy_violations: List[str]
    recommendations: List[str]
    validation_time_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class UnifiedSecurityFramework:
    """
    Unified Security Framework - Production-ready security orchestration.
    
    Integrates all security components into a cohesive system providing:
    - Centralized security policy management
    - Multi-layer threat detection and prevention
    - Real-time monitoring and alerting
    - Compliance validation and reporting
    - Enterprise-grade access controls
    """
    
    def __init__(self, config: SecurityFrameworkConfig = None):
        """Initialize the unified security framework."""
        self.config = config or SecurityFrameworkConfig()
        self.framework_id = str(uuid.uuid4())
        self.status = SecurityFrameworkStatus.INITIALIZING
        self.metrics = SecurityMetrics()
        
        # Initialize security components
        self.advanced_validator = AdvancedSecurityValidator()
        self.api_middleware = APISecurityMiddleware()
        self.security_audit = EnhancedSecurityAudit()
        self.security_safeguards = EnhancedSecuritySafeguards()
        self.enterprise_security = EnterpriseSecuritySystem()
        self.integrated_security = IntegratedSecuritySystem()
        self.monitoring_system = SecurityMonitoringSystem()
        self.policy_engine = SecurityPolicyEngine()
        
        # Observability integration
        self.alerting_system = IntelligentAlertingSystem()
        
        # Security policies registry
        self.active_policies: Dict[SecurityPolicyType, List[Dict]] = {
            policy_type: [] for policy_type in SecurityPolicyType
        }
        
        # Threat intelligence cache
        self.threat_intelligence: Dict[str, Any] = {}
        
        logger.info("Unified security framework initialized", 
                   framework_id=self.framework_id,
                   config=self.config)
    
    async def initialize(self) -> bool:
        """Initialize all security components and validate readiness."""
        try:
            start_time = time.time()
            
            # Initialize components
            await self._initialize_components()
            
            # Load security policies
            await self._load_security_policies()
            
            # Initialize threat intelligence
            await self._initialize_threat_intelligence()
            
            # Validate component integration
            await self._validate_component_integration()
            
            self.status = SecurityFrameworkStatus.OPERATIONAL
            initialization_time = (time.time() - start_time) * 1000
            
            logger.info("Security framework initialization complete",
                       framework_id=self.framework_id,
                       initialization_time_ms=initialization_time,
                       status=self.status.value)
            
            return True
            
        except Exception as e:
            self.status = SecurityFrameworkStatus.DEGRADED
            logger.error("Security framework initialization failed",
                        framework_id=self.framework_id,
                        error=str(e))
            return False
    
    async def comprehensive_security_validation(
        self, 
        request_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> SecurityReport:
        """
        Perform comprehensive security validation across all components.
        
        This is the primary method for validating requests through the
        unified security framework.
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Initialize report
            report = SecurityReport(
                request_id=request_id,
                validation_result=True,
                risk_level=SecurityRiskLevel.LOW,
                threat_categories=[],
                policy_violations=[],
                recommendations=[]
            )
            
            # Advanced threat validation
            if self.config.enable_advanced_validation:
                threat_result = await self.advanced_validator.validate_request(
                    request_data, context
                )
                if not threat_result.is_safe:
                    report.validation_result = False
                    report.risk_level = max(report.risk_level, threat_result.risk_level)
                    report.threat_categories.extend(threat_result.threat_categories)
            
            # API security validation
            if self.config.enable_api_middleware:
                api_result = await self.api_middleware.validate_request(
                    request_data
                )
                if not api_result.allowed:
                    report.validation_result = False
                    report.policy_violations.extend(api_result.violations)
            
            # Enterprise security controls
            if self.config.enable_enterprise_controls:
                enterprise_result = await self.enterprise_security.validate_access(
                    request_data, context
                )
                if not enterprise_result.authorized:
                    report.validation_result = False
                    report.policy_violations.extend(enterprise_result.violations)
            
            # Security safeguards check
            safeguards_result = await self.security_safeguards.evaluate_request(
                request_data, context
            )
            if safeguards_result.control_decision.deny:
                report.validation_result = False
                report.risk_level = max(report.risk_level, safeguards_result.risk_level)
                report.recommendations.extend(safeguards_result.recommendations)
            
            # Update metrics
            self.metrics.total_requests_validated += 1
            if not report.validation_result:
                self.metrics.blocked_requests += 1
            
            # Audit logging
            if self.config.enable_security_audit:
                await self.security_audit.log_security_event(
                    event_type=AuditEventType.SECURITY_VALIDATION,
                    request_id=request_id,
                    result=report.validation_result,
                    details=report.__dict__
                )
            
            # Real-time alerting for high-risk requests
            if report.risk_level in [SecurityRiskLevel.HIGH, SecurityRiskLevel.CRITICAL]:
                await self._trigger_security_alert(report)
            
            # Calculate validation time
            report.validation_time_ms = (time.time() - start_time) * 1000
            self._update_average_validation_time(report.validation_time_ms)
            
            return report
            
        except Exception as e:
            logger.error("Security validation failed",
                        request_id=request_id,
                        error=str(e))
            
            # Return failed validation result
            return SecurityReport(
                request_id=request_id,
                validation_result=False,
                risk_level=SecurityRiskLevel.CRITICAL,
                threat_categories=[ThreatCategory.SYSTEM_ERROR],
                policy_violations=["Security validation system error"],
                recommendations=["Contact security team immediately"],
                validation_time_ms=(time.time() - start_time) * 1000
            )
    
    async def deploy_production_security(self) -> Dict[str, Any]:
        """
        Deploy security framework for production environment.
        
        Configures and validates all security components for production deployment.
        """
        deployment_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            deployment_result = {
                "deployment_id": deployment_id,
                "success": True,
                "components_deployed": [],
                "security_policies_active": 0,
                "monitoring_enabled": False,
                "compliance_validated": False,
                "deployment_time_ms": 0
            }
            
            # Deploy API security middleware
            if self.config.enable_api_middleware:
                await self.api_middleware.deploy_production_config()
                deployment_result["components_deployed"].append("api_security_middleware")
            
            # Deploy enterprise security controls
            if self.config.enable_enterprise_controls:
                await self.enterprise_security.deploy_enterprise_controls()
                deployment_result["components_deployed"].append("enterprise_security_system")
            
            # Activate security policies
            active_policies = await self._activate_production_security_policies()
            deployment_result["security_policies_active"] = len(active_policies)
            
            # Enable security monitoring
            if self.config.enable_threat_monitoring:
                await self.monitoring_system.enable_production_monitoring()
                deployment_result["monitoring_enabled"] = True
            
            # Validate compliance
            compliance_result = await self._validate_production_compliance()
            deployment_result["compliance_validated"] = compliance_result
            
            # Initialize real-time alerting
            if self.config.enable_real_time_alerts:
                await self.alerting_system.initialize_production_alerts()
            
            deployment_result["deployment_time_ms"] = (time.time() - start_time) * 1000
            
            logger.info("Production security deployment complete",
                       deployment_id=deployment_id,
                       result=deployment_result)
            
            return deployment_result
            
        except Exception as e:
            logger.error("Production security deployment failed",
                        deployment_id=deployment_id,
                        error=str(e))
            
            return {
                "deployment_id": deployment_id,
                "success": False,
                "error": str(e),
                "deployment_time_ms": (time.time() - start_time) * 1000
            }
    
    async def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security framework status."""
        return {
            "framework_id": self.framework_id,
            "status": self.status.value,
            "config": self.config.__dict__,
            "metrics": self.metrics.__dict__,
            "active_policies": {
                policy_type.value: len(policies) 
                for policy_type, policies in self.active_policies.items()
            },
            "components": {
                "advanced_validator": self.advanced_validator.get_status(),
                "api_middleware": self.api_middleware.get_status(),
                "security_audit": self.security_audit.get_status(),
                "enterprise_security": self.enterprise_security.get_status(),
                "monitoring_system": self.monitoring_system.get_status()
            }
        }
    
    async def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report for auditing."""
        report_id = str(uuid.uuid4())
        
        try:
            compliance_report = {
                "report_id": report_id,
                "generated_at": datetime.utcnow().isoformat(),
                "framework_status": self.status.value,
                "security_metrics": self.metrics.__dict__,
                "policy_compliance": {},
                "threat_detection_summary": {},
                "audit_trail_summary": {},
                "recommendations": []
            }
            
            # Policy compliance analysis
            for policy_type in SecurityPolicyType:
                compliance_score = await self._calculate_policy_compliance(policy_type)
                compliance_report["policy_compliance"][policy_type.value] = compliance_score
            
            # Threat detection summary
            threat_summary = await self.monitoring_system.get_threat_detection_summary()
            compliance_report["threat_detection_summary"] = threat_summary
            
            # Audit trail summary
            if self.config.enable_security_audit:
                audit_summary = await self.security_audit.get_audit_summary()
                compliance_report["audit_trail_summary"] = audit_summary
            
            # Generate recommendations
            recommendations = await self._generate_security_recommendations()
            compliance_report["recommendations"] = recommendations
            
            return compliance_report
            
        except Exception as e:
            logger.error("Failed to generate compliance report",
                        report_id=report_id,
                        error=str(e))
            return {
                "report_id": report_id,
                "error": str(e),
                "generated_at": datetime.utcnow().isoformat()
            }
    
    # Private helper methods
    
    async def _initialize_components(self):
        """Initialize all security components."""
        components = [
            ("advanced_validator", self.advanced_validator),
            ("api_middleware", self.api_middleware),
            ("security_audit", self.security_audit),
            ("security_safeguards", self.security_safeguards),
            ("enterprise_security", self.enterprise_security),
            ("integrated_security", self.integrated_security),
            ("monitoring_system", self.monitoring_system),
            ("policy_engine", self.policy_engine)
        ]
        
        for name, component in components:
            try:
                await component.initialize()
                logger.debug("Component initialized", component=name)
            except Exception as e:
                logger.error("Component initialization failed", 
                           component=name, error=str(e))
                raise
    
    async def _load_security_policies(self):
        """Load and activate security policies."""
        # Load default policies for each type
        default_policies = {
            SecurityPolicyType.ACCESS_CONTROL: [
                {"name": "admin_access", "level": "high", "enabled": True},
                {"name": "user_access", "level": "medium", "enabled": True}
            ],
            SecurityPolicyType.API_SECURITY: [
                {"name": "rate_limiting", "requests_per_minute": 100, "enabled": True},
                {"name": "input_validation", "strict_mode": True, "enabled": True}
            ],
            SecurityPolicyType.THREAT_PREVENTION: [
                {"name": "sql_injection_prevention", "enabled": True},
                {"name": "xss_protection", "enabled": True}
            ]
        }
        
        for policy_type, policies in default_policies.items():
            self.active_policies[policy_type] = policies
    
    async def _initialize_threat_intelligence(self):
        """Initialize threat intelligence data."""
        self.threat_intelligence = {
            "known_attack_patterns": [],
            "ip_blacklist": [],
            "suspicious_user_agents": [],
            "last_updated": datetime.utcnow().isoformat()
        }
    
    async def _validate_component_integration(self):
        """Validate that all components are properly integrated."""
        # Test communication between components
        test_request = {
            "test_validation": True,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        result = await self.comprehensive_security_validation(test_request)
        
        if not result:
            raise Exception("Component integration validation failed")
    
    async def _trigger_security_alert(self, report: SecurityReport):
        """Trigger security alert for high-risk requests."""
        alert_data = {
            "alert_type": "security_threat_detected",
            "severity": "critical" if report.risk_level == SecurityRiskLevel.CRITICAL else "high",
            "request_id": report.request_id,
            "threat_categories": [cat.value for cat in report.threat_categories],
            "policy_violations": report.policy_violations,
            "timestamp": report.timestamp.isoformat()
        }
        
        await self.alerting_system.trigger_alert(alert_data)
    
    async def _activate_production_security_policies(self) -> List[Dict]:
        """Activate security policies for production deployment."""
        activated_policies = []
        
        for policy_type, policies in self.active_policies.items():
            for policy in policies:
                if policy.get("enabled", False):
                    await self.policy_engine.activate_policy(policy_type, policy)
                    activated_policies.append(policy)
        
        return activated_policies
    
    async def _validate_production_compliance(self) -> bool:
        """Validate compliance for production deployment."""
        try:
            # Check required security components
            required_components = [
                self.advanced_validator,
                self.api_middleware,
                self.enterprise_security,
                self.monitoring_system
            ]
            
            for component in required_components:
                status = await component.health_check()
                if not status.get("healthy", False):
                    return False
            
            # Validate security policies
            for policy_type in SecurityPolicyType:
                if not self.active_policies[policy_type]:
                    logger.warning("No active policies for type", policy_type=policy_type.value)
            
            return True
            
        except Exception as e:
            logger.error("Production compliance validation failed", error=str(e))
            return False
    
    def _update_average_validation_time(self, validation_time_ms: float):
        """Update average validation time metric."""
        if self.metrics.total_requests_validated == 1:
            self.metrics.average_validation_time_ms = validation_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.average_validation_time_ms = (
                alpha * validation_time_ms + 
                (1 - alpha) * self.metrics.average_validation_time_ms
            )
    
    async def _calculate_policy_compliance(self, policy_type: SecurityPolicyType) -> float:
        """Calculate compliance score for a policy type."""
        policies = self.active_policies.get(policy_type, [])
        if not policies:
            return 0.0
        
        enabled_policies = sum(1 for policy in policies if policy.get("enabled", False))
        return (enabled_policies / len(policies)) * 100.0
    
    async def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations based on current state."""
        recommendations = []
        
        # Check validation performance
        if self.metrics.average_validation_time_ms > 100:
            recommendations.append(
                "Consider optimizing security validation performance - "
                f"current average: {self.metrics.average_validation_time_ms:.1f}ms"
            )
        
        # Check blocked requests ratio
        if self.metrics.total_requests_validated > 0:
            block_ratio = self.metrics.blocked_requests / self.metrics.total_requests_validated
            if block_ratio > 0.1:  # More than 10% blocked
                recommendations.append(
                    f"High block ratio ({block_ratio:.1%}) - review security policies"
                )
        
        # Check policy coverage
        for policy_type in SecurityPolicyType:
            if not self.active_policies[policy_type]:
                recommendations.append(
                    f"No active policies for {policy_type.value} - consider adding policies"
                )
        
        return recommendations


# Global instance for production use
_security_framework_instance: Optional[UnifiedSecurityFramework] = None


async def get_security_framework() -> UnifiedSecurityFramework:
    """Get the global security framework instance."""
    global _security_framework_instance
    
    if _security_framework_instance is None:
        _security_framework_instance = UnifiedSecurityFramework()
        await _security_framework_instance.initialize()
    
    return _security_framework_instance


async def validate_request_security(
    request_data: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None
) -> SecurityReport:
    """Convenience function for security validation."""
    framework = await get_security_framework()
    return await framework.comprehensive_security_validation(request_data, context)


async def deploy_production_security() -> Dict[str, Any]:
    """Deploy security framework for production."""
    framework = await get_security_framework()
    return await framework.deploy_production_security()