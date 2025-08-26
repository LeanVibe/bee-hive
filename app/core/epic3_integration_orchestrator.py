"""
Epic 3 Integration Orchestrator for LeanVibe Agent Hive 2.0
===========================================================

Comprehensive integration system that connects Epic 3 Security & Operations
components with Epic 1 orchestrator and Epic 2 testing systems for unified
production-ready deployment.

Epic 3 - Security & Operations: System Integration Excellence
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Callable, Tuple
import structlog

# Epic 3 components
from .unified_security_framework import UnifiedSecurityFramework, SecurityReport
from .secure_deployment_orchestrator import SecureDeploymentOrchestrator, DeploymentResult
from .security_compliance_validator import SecurityComplianceValidator, ComplianceReport
from ..observability.production_observability_orchestrator import ProductionObservabilityOrchestrator
from ..observability.production_monitoring_dashboard import ProductionMonitoringDashboard

# Epic 1 components (orchestrator)
from .orchestration.production_orchestrator import ProductionOrchestrator
from .orchestration.universal_orchestrator import UniversalOrchestrator
from .enhanced_orchestration_engine import EnhancedOrchestrationEngine

# Epic 2 components (testing)
from ..testing.comprehensive_testing_framework import ComprehensiveTestingFramework
from ..testing.performance_testing_suite import PerformanceTestingSuite
from ..testing.integration_testing_engine import IntegrationTestingEngine

from .redis import get_redis_client
from .database import get_async_session

logger = structlog.get_logger()


class IntegrationPhase(Enum):
    """Phases of Epic 3 integration process."""
    INITIALIZATION = "initialization"
    SECURITY_INTEGRATION = "security_integration"
    OBSERVABILITY_INTEGRATION = "observability_integration"
    ORCHESTRATOR_INTEGRATION = "orchestrator_integration"
    TESTING_INTEGRATION = "testing_integration"
    PRODUCTION_VALIDATION = "production_validation"
    DEPLOYMENT_READINESS = "deployment_readiness"
    FINAL_VALIDATION = "final_validation"


class IntegrationStatus(Enum):
    """Status of integration operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIALLY_INTEGRATED = "partially_integrated"
    VALIDATED = "validated"


class ComponentType(Enum):
    """Types of system components being integrated."""
    SECURITY = "security"
    OBSERVABILITY = "observability"
    ORCHESTRATOR = "orchestrator"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"


@dataclass
class IntegrationResult:
    """Result of a component integration operation."""
    integration_id: str
    component_type: ComponentType
    component_name: str
    status: IntegrationStatus
    integration_phase: IntegrationPhase
    success_metrics: Dict[str, Any]
    issues_found: List[str]
    recommendations: List[str]
    integration_time_ms: float
    validation_results: Dict[str, Any]
    dependencies_validated: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SystemIntegrationReport:
    """Comprehensive system integration report."""
    report_id: str
    integration_date: datetime
    overall_integration_status: IntegrationStatus
    components_integrated: List[IntegrationResult]
    epic1_integration_status: IntegrationStatus
    epic2_integration_status: IntegrationStatus
    epic3_deployment_readiness: bool
    production_readiness_score: float
    security_validation_passed: bool
    performance_benchmarks_met: bool
    compliance_requirements_satisfied: bool
    integration_recommendations: List[str]
    next_validation_date: datetime
    total_integration_time_ms: float
    technical_details: Dict[str, Any]


class Epic3IntegrationOrchestrator:
    """
    Epic 3 Integration Orchestrator - System integration excellence.
    
    Provides comprehensive integration capabilities including:
    - Unified security framework integration with Epic 1 & 2
    - Production observability system integration
    - Secure deployment orchestration with existing systems
    - Comprehensive testing integration and validation
    - Production monitoring dashboard integration
    - Cross-epic communication and coordination
    - End-to-end system validation and readiness assessment
    """
    
    def __init__(self):
        """Initialize the Epic 3 integration orchestrator."""
        self.orchestrator_id = str(uuid.uuid4())
        
        # Epic 3 components
        self.security_framework = UnifiedSecurityFramework()
        self.deployment_orchestrator = SecureDeploymentOrchestrator()
        self.compliance_validator = SecurityComplianceValidator()
        self.observability_orchestrator = ProductionObservabilityOrchestrator()
        self.monitoring_dashboard = ProductionMonitoringDashboard()
        
        # Epic 1 components (orchestrator)
        self.production_orchestrator = ProductionOrchestrator()
        self.universal_orchestrator = UniversalOrchestrator()
        self.orchestration_engine = EnhancedOrchestrationEngine()
        
        # Epic 2 components (testing)
        self.testing_framework = ComprehensiveTestingFramework()
        self.performance_testing = PerformanceTestingSuite()
        self.integration_testing = IntegrationTestingEngine()
        
        # Integration state management
        self.integration_results: List[IntegrationResult] = []
        self.integration_status: Dict[ComponentType, IntegrationStatus] = {
            component_type: IntegrationStatus.PENDING for component_type in ComponentType
        }
        
        # Cross-epic communication channels
        self.epic_communication_channels: Dict[str, Any] = {}
        
        # Production readiness metrics
        self.readiness_metrics: Dict[str, float] = {}
        
        logger.info("Epic 3 integration orchestrator initialized",
                   orchestrator_id=self.orchestrator_id)
    
    async def initialize(self) -> bool:
        """Initialize the Epic 3 integration orchestrator."""
        try:
            start_time = time.time()
            
            # Initialize Epic 3 components
            await self._initialize_epic3_components()
            
            # Initialize Epic 1 integration
            await self._initialize_epic1_integration()
            
            # Initialize Epic 2 integration
            await self._initialize_epic2_integration()
            
            # Set up cross-epic communication
            await self._setup_cross_epic_communication()
            
            # Initialize integration monitoring
            await self._setup_integration_monitoring()
            
            initialization_time = (time.time() - start_time) * 1000
            
            logger.info("Epic 3 integration orchestrator initialization complete",
                       orchestrator_id=self.orchestrator_id,
                       initialization_time_ms=initialization_time)
            
            return True
            
        except Exception as e:
            logger.error("Epic 3 integration orchestrator initialization failed",
                        orchestrator_id=self.orchestrator_id,
                        error=str(e))
            return False
    
    async def execute_comprehensive_system_integration(self) -> SystemIntegrationReport:
        """
        Execute comprehensive system integration across all epics.
        
        Primary method for orchestrating complete system integration including
        security, observability, orchestrator, and testing components.
        """
        integration_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            logger.info("Starting comprehensive system integration",
                       integration_id=integration_id)
            
            # Initialize integration report
            integration_results = []
            
            # Phase 1: Security Framework Integration
            security_integration = await self._integrate_security_framework()
            integration_results.append(security_integration)
            self.integration_status[ComponentType.SECURITY] = security_integration.status
            
            # Phase 2: Observability System Integration
            observability_integration = await self._integrate_observability_systems()
            integration_results.append(observability_integration)
            self.integration_status[ComponentType.OBSERVABILITY] = observability_integration.status
            
            # Phase 3: Orchestrator Integration (Epic 1)
            orchestrator_integration = await self._integrate_with_epic1_orchestrator()
            integration_results.append(orchestrator_integration)
            self.integration_status[ComponentType.ORCHESTRATOR] = orchestrator_integration.status
            
            # Phase 4: Testing System Integration (Epic 2)
            testing_integration = await self._integrate_with_epic2_testing()
            integration_results.append(testing_integration)
            self.integration_status[ComponentType.TESTING] = testing_integration.status
            
            # Phase 5: Deployment System Integration
            deployment_integration = await self._integrate_deployment_systems()
            integration_results.append(deployment_integration)
            self.integration_status[ComponentType.DEPLOYMENT] = deployment_integration.status
            
            # Phase 6: Monitoring Dashboard Integration
            monitoring_integration = await self._integrate_monitoring_dashboards()
            integration_results.append(monitoring_integration)
            self.integration_status[ComponentType.MONITORING] = monitoring_integration.status
            
            # Phase 7: Cross-Epic Validation
            validation_results = await self._execute_cross_epic_validation()
            
            # Phase 8: Production Readiness Assessment
            readiness_assessment = await self._assess_production_readiness()
            
            # Determine overall integration status
            overall_status = self._determine_overall_integration_status(integration_results)
            
            # Calculate production readiness score
            production_readiness_score = self._calculate_production_readiness_score(
                integration_results, validation_results, readiness_assessment
            )
            
            # Generate integration recommendations
            integration_recommendations = self._generate_integration_recommendations(
                integration_results, validation_results
            )
            
            # Create comprehensive integration report
            integration_report = SystemIntegrationReport(
                report_id=integration_id,
                integration_date=datetime.utcnow(),
                overall_integration_status=overall_status,
                components_integrated=integration_results,
                epic1_integration_status=self.integration_status[ComponentType.ORCHESTRATOR],
                epic2_integration_status=self.integration_status[ComponentType.TESTING],
                epic3_deployment_readiness=production_readiness_score >= 80.0,
                production_readiness_score=production_readiness_score,
                security_validation_passed=validation_results.get("security_passed", False),
                performance_benchmarks_met=validation_results.get("performance_passed", False),
                compliance_requirements_satisfied=validation_results.get("compliance_passed", False),
                integration_recommendations=integration_recommendations,
                next_validation_date=datetime.utcnow() + timedelta(days=7),
                total_integration_time_ms=(time.time() - start_time) * 1000,
                technical_details={
                    "components_integrated": len(integration_results),
                    "successful_integrations": sum(1 for r in integration_results if r.status == IntegrationStatus.SUCCESS),
                    "failed_integrations": sum(1 for r in integration_results if r.status == IntegrationStatus.FAILED),
                    "integration_methodology": "Epic 3 comprehensive integration orchestration"
                }
            )
            
            # Store integration results
            self.integration_results.extend(integration_results)
            
            logger.info("Comprehensive system integration completed",
                       integration_id=integration_id,
                       overall_status=overall_status.value,
                       production_readiness_score=production_readiness_score,
                       total_time_ms=integration_report.total_integration_time_ms)
            
            return integration_report
            
        except Exception as e:
            logger.error("Comprehensive system integration failed",
                        integration_id=integration_id,
                        error=str(e))
            
            # Return error report
            return SystemIntegrationReport(
                report_id=integration_id,
                integration_date=datetime.utcnow(),
                overall_integration_status=IntegrationStatus.FAILED,
                components_integrated=[],
                epic1_integration_status=IntegrationStatus.FAILED,
                epic2_integration_status=IntegrationStatus.FAILED,
                epic3_deployment_readiness=False,
                production_readiness_score=0.0,
                security_validation_passed=False,
                performance_benchmarks_met=False,
                compliance_requirements_satisfied=False,
                integration_recommendations=[f"Critical: Fix integration system - {str(e)}"],
                next_validation_date=datetime.utcnow() + timedelta(days=1),
                total_integration_time_ms=(time.time() - start_time) * 1000,
                technical_details={"error": str(e)}
            )
    
    async def validate_epic1_orchestrator_integration(self) -> Dict[str, Any]:
        """Validate integration with Epic 1 orchestrator systems."""
        validation_id = str(uuid.uuid4())
        
        try:
            logger.info("Validating Epic 1 orchestrator integration",
                       validation_id=validation_id)
            
            validation_results = {
                "validation_id": validation_id,
                "epic1_orchestrator_accessible": False,
                "security_framework_integrated": False,
                "observability_connected": False,
                "deployment_coordination_working": False,
                "performance_metrics_shared": False,
                "overall_integration_health": "unknown"
            }
            
            # Test Epic 1 orchestrator connectivity
            try:
                orchestrator_status = await self.production_orchestrator.get_system_status()
                validation_results["epic1_orchestrator_accessible"] = orchestrator_status.get("healthy", False)
            except Exception as e:
                logger.error("Epic 1 orchestrator connectivity test failed", error=str(e))
            
            # Test security framework integration
            try:
                security_report = await self.security_framework.comprehensive_security_validation({
                    "test_integration": True,
                    "epic1_validation": True
                })
                validation_results["security_framework_integrated"] = security_report.validation_result
            except Exception as e:
                logger.error("Security framework integration test failed", error=str(e))
            
            # Test observability integration
            try:
                observability_data = await self.observability_orchestrator.collect_comprehensive_metrics()
                validation_results["observability_connected"] = "error" not in observability_data
            except Exception as e:
                logger.error("Observability integration test failed", error=str(e))
            
            # Test deployment coordination
            try:
                deployment_status = await self.deployment_orchestrator.get_deployment_status("test-integration")
                validation_results["deployment_coordination_working"] = deployment_status is not None
            except Exception as e:
                logger.error("Deployment coordination test failed", error=str(e))
            
            # Calculate overall integration health
            passed_tests = sum(1 for key, value in validation_results.items() 
                             if key.endswith(("_accessible", "_integrated", "_connected", "_working")) and value)
            total_tests = 4
            health_percentage = (passed_tests / total_tests) * 100
            
            if health_percentage >= 90:
                validation_results["overall_integration_health"] = "excellent"
            elif health_percentage >= 70:
                validation_results["overall_integration_health"] = "good"
            elif health_percentage >= 50:
                validation_results["overall_integration_health"] = "fair"
            else:
                validation_results["overall_integration_health"] = "poor"
            
            validation_results["integration_health_percentage"] = health_percentage
            validation_results["tests_passed"] = passed_tests
            validation_results["total_tests"] = total_tests
            
            return validation_results
            
        except Exception as e:
            logger.error("Epic 1 orchestrator integration validation failed",
                        validation_id=validation_id,
                        error=str(e))
            return {
                "validation_id": validation_id,
                "error": str(e),
                "overall_integration_health": "failed"
            }
    
    async def validate_epic2_testing_integration(self) -> Dict[str, Any]:
        """Validate integration with Epic 2 testing systems."""
        validation_id = str(uuid.uuid4())
        
        try:
            logger.info("Validating Epic 2 testing integration",
                       validation_id=validation_id)
            
            validation_results = {
                "validation_id": validation_id,
                "testing_framework_accessible": False,
                "performance_testing_integrated": False,
                "security_testing_coordinated": False,
                "integration_tests_executable": False,
                "test_results_reportable": False,
                "overall_testing_health": "unknown"
            }
            
            # Test Epic 2 testing framework connectivity
            try:
                testing_status = await self.testing_framework.get_framework_status()
                validation_results["testing_framework_accessible"] = testing_status.get("operational", False)
            except Exception as e:
                logger.error("Epic 2 testing framework connectivity test failed", error=str(e))
            
            # Test performance testing integration
            try:
                performance_config = await self.performance_testing.get_test_configuration()
                validation_results["performance_testing_integrated"] = performance_config is not None
            except Exception as e:
                logger.error("Performance testing integration test failed", error=str(e))
            
            # Test security testing coordination
            try:
                security_test_result = await self._test_security_testing_coordination()
                validation_results["security_testing_coordinated"] = security_test_result
            except Exception as e:
                logger.error("Security testing coordination test failed", error=str(e))
            
            # Test integration test execution
            try:
                integration_test_result = await self.integration_testing.execute_test_suite({
                    "test_type": "integration_validation",
                    "scope": "epic3_integration"
                })
                validation_results["integration_tests_executable"] = integration_test_result.get("success", False)
            except Exception as e:
                logger.error("Integration test execution failed", error=str(e))
            
            # Test reporting capabilities
            try:
                test_report = await self._generate_epic2_integration_report()
                validation_results["test_results_reportable"] = test_report is not None
            except Exception as e:
                logger.error("Test reporting validation failed", error=str(e))
            
            # Calculate overall testing health
            passed_tests = sum(1 for key, value in validation_results.items() 
                             if key.endswith(("_accessible", "_integrated", "_coordinated", "_executable", "_reportable")) and value)
            total_tests = 5
            health_percentage = (passed_tests / total_tests) * 100
            
            if health_percentage >= 90:
                validation_results["overall_testing_health"] = "excellent"
            elif health_percentage >= 70:
                validation_results["overall_testing_health"] = "good"
            elif health_percentage >= 50:
                validation_results["overall_testing_health"] = "fair"
            else:
                validation_results["overall_testing_health"] = "poor"
            
            validation_results["testing_health_percentage"] = health_percentage
            validation_results["tests_passed"] = passed_tests
            validation_results["total_tests"] = total_tests
            
            return validation_results
            
        except Exception as e:
            logger.error("Epic 2 testing integration validation failed",
                        validation_id=validation_id,
                        error=str(e))
            return {
                "validation_id": validation_id,
                "error": str(e),
                "overall_testing_health": "failed"
            }
    
    async def get_system_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive system integration status."""
        status_id = str(uuid.uuid4())
        
        try:
            # Get component statuses
            component_statuses = {}
            for component_type in ComponentType:
                component_statuses[component_type.value] = self.integration_status[component_type].value
            
            # Get recent integration results
            recent_results = self.integration_results[-10:] if self.integration_results else []
            
            # Calculate integration health metrics
            successful_integrations = sum(1 for r in recent_results if r.status == IntegrationStatus.SUCCESS)
            total_integrations = len(recent_results)
            integration_success_rate = (successful_integrations / total_integrations * 100) if total_integrations > 0 else 0
            
            # Get Epic 1 & 2 integration health
            epic1_health = await self.validate_epic1_orchestrator_integration()
            epic2_health = await self.validate_epic2_testing_integration()
            
            # Calculate overall system readiness
            overall_readiness = self._calculate_overall_system_readiness(
                component_statuses, epic1_health, epic2_health
            )
            
            status_report = {
                "status_id": status_id,
                "generated_at": datetime.utcnow().isoformat(),
                "overall_system_readiness": overall_readiness,
                "component_integration_status": component_statuses,
                "integration_success_rate": integration_success_rate,
                "epic1_integration_health": {
                    "status": epic1_health.get("overall_integration_health", "unknown"),
                    "health_percentage": epic1_health.get("integration_health_percentage", 0)
                },
                "epic2_integration_health": {
                    "status": epic2_health.get("overall_testing_health", "unknown"),
                    "health_percentage": epic2_health.get("testing_health_percentage", 0)
                },
                "recent_integrations": [
                    {
                        "component": result.component_name,
                        "status": result.status.value,
                        "timestamp": result.timestamp.isoformat()
                    }
                    for result in recent_results
                ],
                "readiness_metrics": self.readiness_metrics,
                "next_actions": self._get_next_integration_actions()
            }
            
            return status_report
            
        except Exception as e:
            logger.error("Failed to get system integration status",
                        status_id=status_id,
                        error=str(e))
            return {
                "status_id": status_id,
                "error": str(e),
                "generated_at": datetime.utcnow().isoformat()
            }
    
    # Private helper methods
    
    async def _initialize_epic3_components(self):
        """Initialize all Epic 3 components."""
        components = [
            ("security_framework", self.security_framework),
            ("deployment_orchestrator", self.deployment_orchestrator),
            ("compliance_validator", self.compliance_validator),
            ("observability_orchestrator", self.observability_orchestrator),
            ("monitoring_dashboard", self.monitoring_dashboard)
        ]
        
        for name, component in components:
            try:
                await component.initialize()
                logger.debug("Epic 3 component initialized", component=name)
            except Exception as e:
                logger.error("Epic 3 component initialization failed", 
                           component=name, error=str(e))
                raise
    
    async def _initialize_epic1_integration(self):
        """Initialize integration with Epic 1 systems."""
        try:
            # Initialize Epic 1 orchestrator components
            await self.production_orchestrator.initialize()
            await self.universal_orchestrator.initialize()
            await self.orchestration_engine.initialize()
            
            # Set up communication channels
            self.epic_communication_channels["epic1"] = {
                "production_orchestrator": self.production_orchestrator,
                "universal_orchestrator": self.universal_orchestrator,
                "orchestration_engine": self.orchestration_engine
            }
            
            logger.info("Epic 1 integration initialized")
            
        except Exception as e:
            logger.error("Epic 1 integration initialization failed", error=str(e))
            raise
    
    async def _initialize_epic2_integration(self):
        """Initialize integration with Epic 2 systems."""
        try:
            # Initialize Epic 2 testing components
            await self.testing_framework.initialize()
            await self.performance_testing.initialize()
            await self.integration_testing.initialize()
            
            # Set up communication channels
            self.epic_communication_channels["epic2"] = {
                "testing_framework": self.testing_framework,
                "performance_testing": self.performance_testing,
                "integration_testing": self.integration_testing
            }
            
            logger.info("Epic 2 integration initialized")
            
        except Exception as e:
            logger.error("Epic 2 integration initialization failed", error=str(e))
            raise
    
    async def _integrate_security_framework(self) -> IntegrationResult:
        """Integrate unified security framework."""
        integration_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Deploy production security
            deployment_result = await self.security_framework.deploy_production_security()
            
            # Validate security integration with Epic 1
            epic1_security_validation = await self._validate_epic1_security_integration()
            
            # Validate security integration with Epic 2
            epic2_security_validation = await self._validate_epic2_security_integration()
            
            success = (
                deployment_result.get("success", False) and
                epic1_security_validation and
                epic2_security_validation
            )
            
            return IntegrationResult(
                integration_id=integration_id,
                component_type=ComponentType.SECURITY,
                component_name="unified_security_framework",
                status=IntegrationStatus.SUCCESS if success else IntegrationStatus.FAILED,
                integration_phase=IntegrationPhase.SECURITY_INTEGRATION,
                success_metrics={
                    "deployment_success": deployment_result.get("success", False),
                    "epic1_integration": epic1_security_validation,
                    "epic2_integration": epic2_security_validation,
                    "components_deployed": len(deployment_result.get("components_deployed", []))
                },
                issues_found=[],
                recommendations=["Monitor security framework performance"],
                integration_time_ms=(time.time() - start_time) * 1000,
                validation_results=deployment_result,
                dependencies_validated=["redis", "database", "epic1_orchestrator", "epic2_testing"]
            )
            
        except Exception as e:
            logger.error("Security framework integration failed", error=str(e))
            return IntegrationResult(
                integration_id=integration_id,
                component_type=ComponentType.SECURITY,
                component_name="unified_security_framework",
                status=IntegrationStatus.FAILED,
                integration_phase=IntegrationPhase.SECURITY_INTEGRATION,
                success_metrics={},
                issues_found=[str(e)],
                recommendations=["Fix security framework integration"],
                integration_time_ms=(time.time() - start_time) * 1000,
                validation_results={},
                dependencies_validated=[]
            )
    
    async def _integrate_observability_systems(self) -> IntegrationResult:
        """Integrate production observability systems."""
        integration_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Deploy production observability
            deployment_result = await self.observability_orchestrator.deploy_production_monitoring()
            
            # Initialize monitoring dashboard
            dashboard_result = await self.monitoring_dashboard.initialize()
            
            success = deployment_result.get("success", False) and dashboard_result
            
            return IntegrationResult(
                integration_id=integration_id,
                component_type=ComponentType.OBSERVABILITY,
                component_name="production_observability_orchestrator",
                status=IntegrationStatus.SUCCESS if success else IntegrationStatus.FAILED,
                integration_phase=IntegrationPhase.OBSERVABILITY_INTEGRATION,
                success_metrics={
                    "monitoring_deployment_success": deployment_result.get("success", False),
                    "dashboard_initialized": dashboard_result,
                    "components_deployed": len(deployment_result.get("components_deployed", [])),
                    "monitoring_enabled": deployment_result.get("monitoring_enabled", False)
                },
                issues_found=[],
                recommendations=["Configure alert thresholds", "Set up dashboard access"],
                integration_time_ms=(time.time() - start_time) * 1000,
                validation_results=deployment_result,
                dependencies_validated=["prometheus", "redis", "websocket_streaming"]
            )
            
        except Exception as e:
            logger.error("Observability systems integration failed", error=str(e))
            return IntegrationResult(
                integration_id=integration_id,
                component_type=ComponentType.OBSERVABILITY,
                component_name="production_observability_orchestrator",
                status=IntegrationStatus.FAILED,
                integration_phase=IntegrationPhase.OBSERVABILITY_INTEGRATION,
                success_metrics={},
                issues_found=[str(e)],
                recommendations=["Fix observability integration"],
                integration_time_ms=(time.time() - start_time) * 1000,
                validation_results={},
                dependencies_validated=[]
            )
    
    async def _calculate_production_readiness_score(
        self,
        integration_results: List[IntegrationResult],
        validation_results: Dict[str, Any],
        readiness_assessment: Dict[str, Any]
    ) -> float:
        """Calculate overall production readiness score."""
        # Integration success rate (40% weight)
        successful_integrations = sum(1 for r in integration_results if r.status == IntegrationStatus.SUCCESS)
        integration_score = (successful_integrations / len(integration_results)) * 40 if integration_results else 0
        
        # Security validation (25% weight)
        security_score = 25 if validation_results.get("security_passed", False) else 0
        
        # Performance benchmarks (20% weight)
        performance_score = 20 if validation_results.get("performance_passed", False) else 0
        
        # Compliance requirements (15% weight)
        compliance_score = 15 if validation_results.get("compliance_passed", False) else 0
        
        total_score = integration_score + security_score + performance_score + compliance_score
        
        return min(100.0, max(0.0, total_score))


# Global instance for production use
_epic3_integration_orchestrator_instance: Optional[Epic3IntegrationOrchestrator] = None


async def get_epic3_integration_orchestrator() -> Epic3IntegrationOrchestrator:
    """Get the global Epic 3 integration orchestrator instance."""
    global _epic3_integration_orchestrator_instance
    
    if _epic3_integration_orchestrator_instance is None:
        _epic3_integration_orchestrator_instance = Epic3IntegrationOrchestrator()
        await _epic3_integration_orchestrator_instance.initialize()
    
    return _epic3_integration_orchestrator_instance


async def execute_system_integration() -> SystemIntegrationReport:
    """Convenience function for comprehensive system integration."""
    orchestrator = await get_epic3_integration_orchestrator()
    return await orchestrator.execute_comprehensive_system_integration()


async def validate_epic1_integration() -> Dict[str, Any]:
    """Convenience function for Epic 1 integration validation."""
    orchestrator = await get_epic3_integration_orchestrator()
    return await orchestrator.validate_epic1_orchestrator_integration()


async def validate_epic2_integration() -> Dict[str, Any]:
    """Convenience function for Epic 2 integration validation."""
    orchestrator = await get_epic3_integration_orchestrator()
    return await orchestrator.validate_epic2_testing_integration()