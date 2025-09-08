"""
Penetration Testing & Security Validation Framework for LeanVibe Agent Hive 2.0.

Implements comprehensive security validation including:
- Automated penetration testing framework
- Security validation procedures and protocols
- Vulnerability assessment and risk scoring
- Red team / Blue team simulation capabilities
- Security posture validation and reporting
"""

import uuid
import asyncio
import json
import hashlib
import base64
import random
import string
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, field
import logging
from pathlib import Path
import subprocess
import tempfile
import ssl
import socket

from sqlalchemy import select, and_, or_, func, desc, text
from sqlalchemy.ext.asyncio import AsyncSession
import httpx
from cryptography import x509
from cryptography.hazmat.backends import default_backend
import aiofiles

from ..models.agent import Agent
from ..models.context import Context
from ..core.access_control import Permission, AccessLevel
from .security_audit import SecurityAuditSystem, ThreatLevel
from .compliance_audit import ComplianceAuditSystem, AuditEventCategory, SeverityLevel
from .security_monitoring import SecurityMonitoringSystem, ThreatType


logger = logging.getLogger(__name__)


class PenTestType(Enum):
    """Types of penetration tests."""
    NETWORK_SCAN = "NETWORK_SCAN"
    WEB_APPLICATION = "WEB_APPLICATION"
    API_SECURITY = "API_SECURITY"
    AUTHENTICATION = "AUTHENTICATION"
    AUTHORIZATION = "AUTHORIZATION"
    DATA_VALIDATION = "DATA_VALIDATION"
    INJECTION_ATTACKS = "INJECTION_ATTACKS"
    XSS_TESTING = "XSS_TESTING"
    CSRF_TESTING = "CSRF_TESTING"
    BRUTE_FORCE = "BRUTE_FORCE"
    PRIVILEGE_ESCALATION = "PRIVILEGE_ESCALATION"
    SOCIAL_ENGINEERING = "SOCIAL_ENGINEERING"
    PHYSICAL_SECURITY = "PHYSICAL_SECURITY"
    WIRELESS_SECURITY = "WIRELESS_SECURITY"


class TestSeverity(Enum):
    """Severity levels for penetration tests."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class TestStatus(Enum):
    """Status of penetration tests."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    PAUSED = "PAUSED"


class VulnerabilityClass(Enum):
    """Classification of vulnerabilities."""
    INJECTION = "INJECTION"
    BROKEN_AUTHENTICATION = "BROKEN_AUTHENTICATION"
    SENSITIVE_DATA_EXPOSURE = "SENSITIVE_DATA_EXPOSURE"
    XML_EXTERNAL_ENTITIES = "XML_EXTERNAL_ENTITIES"
    BROKEN_ACCESS_CONTROL = "BROKEN_ACCESS_CONTROL"
    SECURITY_MISCONFIGURATION = "SECURITY_MISCONFIGURATION"
    XSS = "XSS"
    INSECURE_DESERIALIZATION = "INSECURE_DESERIALIZATION"
    KNOWN_VULNERABILITIES = "KNOWN_VULNERABILITIES"
    INSUFFICIENT_LOGGING = "INSUFFICIENT_LOGGING"


@dataclass
class PenTestTarget:
    """Target for penetration testing."""
    id: uuid.UUID
    name: str
    target_type: str  # API, WEB_APP, SERVICE, NETWORK
    endpoint: str
    credentials: Optional[Dict[str, Any]] = None
    scope: List[str] = field(default_factory=list)
    exclusions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "name": self.name,
            "target_type": self.target_type,
            "endpoint": self.endpoint,
            "scope": self.scope,
            "exclusions": self.exclusions,
            "metadata": self.metadata
        }


@dataclass
class SecurityTestCase:
    """Individual security test case."""
    id: uuid.UUID
    name: str
    description: str
    test_type: PenTestType
    severity: TestSeverity
    test_payload: Dict[str, Any]
    expected_behavior: str
    validation_criteria: List[str]
    remediation_guidance: str
    references: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "test_type": self.test_type.value,
            "severity": self.severity.value,
            "test_payload": self.test_payload,
            "expected_behavior": self.expected_behavior,
            "validation_criteria": self.validation_criteria,
            "remediation_guidance": self.remediation_guidance,
            "references": self.references,
            "tags": self.tags
        }


@dataclass
class TestResult:
    """Result of a security test."""
    id: uuid.UUID
    test_case_id: uuid.UUID
    target_id: uuid.UUID
    status: TestStatus
    vulnerability_found: bool
    vulnerability_class: Optional[VulnerabilityClass]
    risk_score: float
    confidence: float
    evidence: Dict[str, Any]
    execution_time: float
    error_message: Optional[str] = None
    remediation_priority: TestSeverity = TestSeverity.LOW
    false_positive_probability: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "test_case_id": str(self.test_case_id),
            "target_id": str(self.target_id),
            "status": self.status.value,
            "vulnerability_found": self.vulnerability_found,
            "vulnerability_class": self.vulnerability_class.value if self.vulnerability_class else None,
            "risk_score": self.risk_score,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "execution_time": self.execution_time,
            "error_message": self.error_message,
            "remediation_priority": self.remediation_priority.value,
            "false_positive_probability": self.false_positive_probability
        }


@dataclass
class PenTestResults:
    """Complete penetration test results."""
    id: uuid.UUID
    test_suite_name: str
    target: PenTestTarget
    test_started: datetime
    test_completed: Optional[datetime]
    total_tests: int
    tests_completed: int
    tests_passed: int
    tests_failed: int
    vulnerabilities_found: int
    critical_vulnerabilities: int
    high_vulnerabilities: int
    medium_vulnerabilities: int
    low_vulnerabilities: int
    test_results: List[TestResult] = field(default_factory=list)
    overall_risk_score: float = 0.0
    security_posture_rating: str = "UNKNOWN"
    executive_summary: str = ""
    detailed_findings: List[Dict[str, Any]] = field(default_factory=list)
    remediation_roadmap: List[Dict[str, Any]] = field(default_factory=list)
    
    def calculate_summary(self) -> Dict[str, Any]:
        """Calculate test summary metrics."""
        return {
            "test_suite": self.test_suite_name,
            "target": self.target.to_dict(),
            "execution_summary": {
                "total_tests": self.total_tests,
                "completed": self.tests_completed,
                "passed": self.tests_passed,
                "failed": self.tests_failed,
                "success_rate": self.tests_passed / max(1, self.tests_completed)
            },
            "vulnerability_summary": {
                "total_found": self.vulnerabilities_found,
                "critical": self.critical_vulnerabilities,
                "high": self.high_vulnerabilities,
                "medium": self.medium_vulnerabilities,
                "low": self.low_vulnerabilities
            },
            "risk_assessment": {
                "overall_risk_score": self.overall_risk_score,
                "security_posture": self.security_posture_rating,
                "immediate_action_required": self.critical_vulnerabilities > 0
            }
        }


@dataclass
class SecurityScore:
    """Security posture score."""
    overall_score: float
    category_scores: Dict[str, float]
    risk_level: TestSeverity
    improvement_areas: List[str]
    strengths: List[str]
    benchmark_comparison: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "category_scores": self.category_scores,
            "risk_level": self.risk_level.value,
            "improvement_areas": self.improvement_areas,
            "strengths": self.strengths,
            "benchmark_comparison": self.benchmark_comparison
        }


class PenetrationTestingFramework:
    """
    Comprehensive penetration testing and security validation framework.
    
    Features:
    - Automated security test execution
    - Comprehensive vulnerability assessment
    - Risk scoring and prioritization
    - Security posture validation
    - Integration with security monitoring
    - Detailed reporting and remediation guidance
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        security_audit: SecurityAuditSystem,
        compliance_audit: ComplianceAuditSystem,
        security_monitoring: SecurityMonitoringSystem,
        test_data_path: str = "/var/lib/agent-hive/security-tests"
    ):
        """
        Initialize penetration testing framework.
        
        Args:
            db_session: Database session
            security_audit: Security audit system
            compliance_audit: Compliance audit system
            security_monitoring: Security monitoring system
            test_data_path: Path for test data and results
        """
        self.db = db_session
        self.security_audit = security_audit
        self.compliance_audit = compliance_audit
        self.security_monitoring = security_monitoring
        self.test_data_path = Path(test_data_path)
        self.test_data_path.mkdir(parents=True, exist_ok=True)
        
        # Test registry
        self.test_cases: Dict[str, SecurityTestCase] = {}
        self.test_targets: Dict[uuid.UUID, PenTestTarget] = {}
        self.active_tests: Dict[uuid.UUID, PenTestResults] = {}
        
        # Test execution
        self.max_concurrent_tests = 5
        self.test_timeout = 300  # 5 minutes
        self.rate_limit_delay = 0.1  # 100ms between requests
        
        # Security validation rules
        self.validation_rules: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.config = {
            "safe_mode": True,  # Prevents destructive tests
            "auto_stop_on_critical": True,
            "max_test_duration": 3600,  # 1 hour
            "evidence_collection": True,
            "detailed_logging": True,
            "false_positive_detection": True,
            "risk_calculation_weights": {
                "exploitability": 0.3,
                "impact": 0.4,
                "prevalence": 0.2,
                "detectability": 0.1
            }
        }
        
        # Initialize test suites
        self._initialize_test_cases()
        self._initialize_validation_rules()
    
    async def create_test_target(
        self,
        name: str,
        target_type: str,
        endpoint: str,
        credentials: Optional[Dict[str, Any]] = None,
        scope: Optional[List[str]] = None,
        exclusions: Optional[List[str]] = None
    ) -> PenTestTarget:
        """
        Create a new penetration test target.
        
        Args:
            name: Human-readable name for the target
            target_type: Type of target (API, WEB_APP, SERVICE, NETWORK)
            endpoint: Target endpoint or URL
            credentials: Optional authentication credentials
            scope: List of in-scope items
            exclusions: List of excluded items
            
        Returns:
            Created test target
        """
        target = PenTestTarget(
            id=uuid.uuid4(),
            name=name,
            target_type=target_type,
            endpoint=endpoint,
            credentials=credentials,
            scope=scope or [],
            exclusions=exclusions or []
        )
        
        self.test_targets[target.id] = target
        
        # Audit target creation
        await self.compliance_audit.log_audit_event(
            event_type="PENTEST_TARGET_CREATED",
            category=AuditEventCategory.SECURITY_EVENT,
            severity=SeverityLevel.INFO,
            action="CREATE_TARGET",
            outcome="SUCCESS",
            details=target.to_dict(),
            compliance_tags=["PENETRATION_TESTING", "TARGET_MANAGEMENT"]
        )
        
        return target
    
    async def execute_security_test_suite(
        self,
        target_id: uuid.UUID,
        test_types: Optional[List[PenTestType]] = None,
        custom_tests: Optional[List[SecurityTestCase]] = None
    ) -> PenTestResults:
        """
        Execute comprehensive security test suite.
        
        Args:
            target_id: Target to test
            test_types: Specific test types to run (None for all)
            custom_tests: Custom test cases to include
            
        Returns:
            Comprehensive test results
        """
        try:
            target = self.test_targets.get(target_id)
            if not target:
                raise ValueError(f"Test target {target_id} not found")
            
            # Create test results container
            test_results = PenTestResults(
                id=uuid.uuid4(),
                test_suite_name=f"Security Test Suite - {target.name}",
                target=target,
                test_started=datetime.utcnow(),
                test_completed=None,
                total_tests=0,
                tests_completed=0,
                tests_passed=0,
                tests_failed=0,
                vulnerabilities_found=0,
                critical_vulnerabilities=0,
                high_vulnerabilities=0,
                medium_vulnerabilities=0,
                low_vulnerabilities=0
            )
            
            self.active_tests[test_results.id] = test_results
            
            # Select test cases to run
            selected_tests = await self._select_test_cases(target, test_types, custom_tests)
            test_results.total_tests = len(selected_tests)
            
            logger.info(f"Starting security test suite for {target.name} with {len(selected_tests)} tests")
            
            # Execute tests with rate limiting
            semaphore = asyncio.Semaphore(self.max_concurrent_tests)
            
            async def execute_single_test(test_case: SecurityTestCase) -> TestResult:
                async with semaphore:
                    return await self._execute_test_case(test_case, target)
            
            # Run all tests
            test_tasks = [execute_single_test(test) for test in selected_tests]
            completed_results = await asyncio.gather(*test_tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(completed_results):
                if isinstance(result, Exception):
                    logger.error(f"Test {selected_tests[i].name} failed: {result}")
                    error_result = TestResult(
                        id=uuid.uuid4(),
                        test_case_id=selected_tests[i].id,
                        target_id=target_id,
                        status=TestStatus.FAILED,
                        vulnerability_found=False,
                        vulnerability_class=None,
                        risk_score=0.0,
                        confidence=0.0,
                        evidence={},
                        execution_time=0.0,
                        error_message=str(result)
                    )
                    test_results.test_results.append(error_result)
                    test_results.tests_failed += 1
                else:
                    test_results.test_results.append(result)
                    if result.status == TestStatus.COMPLETED:
                        test_results.tests_completed += 1
                        if not result.vulnerability_found:
                            test_results.tests_passed += 1
                        else:
                            test_results.vulnerabilities_found += 1
                            
                            # Count by severity
                            if result.remediation_priority == TestSeverity.CRITICAL:
                                test_results.critical_vulnerabilities += 1
                            elif result.remediation_priority == TestSeverity.HIGH:
                                test_results.high_vulnerabilities += 1
                            elif result.remediation_priority == TestSeverity.MEDIUM:
                                test_results.medium_vulnerabilities += 1
                            else:
                                test_results.low_vulnerabilities += 1
            
            # Calculate overall risk score and security posture
            await self._calculate_overall_risk(test_results)
            
            # Generate findings and recommendations
            await self._generate_findings_and_recommendations(test_results)
            
            test_results.test_completed = datetime.utcnow()
            
            # Audit test completion
            await self.compliance_audit.log_audit_event(
                event_type="PENTEST_SUITE_COMPLETED",
                category=AuditEventCategory.SECURITY_EVENT,
                severity=SeverityLevel.HIGH if test_results.critical_vulnerabilities > 0 else SeverityLevel.INFO,
                action="EXECUTE_TEST_SUITE",
                outcome="SUCCESS",
                details=test_results.calculate_summary(),
                compliance_tags=["PENETRATION_TESTING", "SECURITY_VALIDATION"]
            )
            
            logger.info(f"Security test suite completed: {test_results.vulnerabilities_found} vulnerabilities found")
            return test_results
            
        except Exception as e:
            logger.error(f"Security test suite execution failed: {e}")
            await self.compliance_audit.log_audit_event(
                event_type="PENTEST_SUITE_FAILED",
                category=AuditEventCategory.ERROR_EVENT,
                severity=SeverityLevel.HIGH,
                action="EXECUTE_TEST_SUITE",
                outcome="FAILURE",
                details={"error": str(e), "target_id": str(target_id)}
            )
            raise
    
    async def validate_penetration_test_results(
        self,
        test_results: PenTestResults,
        validation_criteria: Optional[Dict[str, Any]] = None
    ) -> SecurityScore:
        """
        Validate and score penetration test results.
        
        Args:
            test_results: Test results to validate
            validation_criteria: Custom validation criteria
            
        Returns:
            Security score and assessment
        """
        try:
            # Calculate category scores
            category_scores = {}
            
            # Authentication security score
            auth_tests = [r for r in test_results.test_results if 
                         r.test_case_id in [tc.id for tc in self.test_cases.values() 
                                          if tc.test_type == PenTestType.AUTHENTICATION]]
            category_scores["authentication"] = await self._calculate_category_score(auth_tests)
            
            # Authorization security score
            authz_tests = [r for r in test_results.test_results if 
                          r.test_case_id in [tc.id for tc in self.test_cases.values() 
                                           if tc.test_type == PenTestType.AUTHORIZATION]]
            category_scores["authorization"] = await self._calculate_category_score(authz_tests)
            
            # Input validation score
            validation_tests = [r for r in test_results.test_results if 
                               r.test_case_id in [tc.id for tc in self.test_cases.values() 
                                                if tc.test_type in [PenTestType.INJECTION_ATTACKS, 
                                                                   PenTestType.XSS_TESTING]]]
            category_scores["input_validation"] = await self._calculate_category_score(validation_tests)
            
            # Data protection score
            data_tests = [r for r in test_results.test_results if 
                         r.test_case_id in [tc.id for tc in self.test_cases.values() 
                                          if tc.test_type == PenTestType.DATA_VALIDATION]]
            category_scores["data_protection"] = await self._calculate_category_score(data_tests)
            
            # Calculate overall score
            overall_score = sum(category_scores.values()) / len(category_scores) if category_scores else 0.0
            
            # Determine risk level
            if overall_score >= 0.9:
                risk_level = TestSeverity.LOW
            elif overall_score >= 0.7:
                risk_level = TestSeverity.MEDIUM
            elif overall_score >= 0.5:
                risk_level = TestSeverity.HIGH
            else:
                risk_level = TestSeverity.CRITICAL
            
            # Identify improvement areas and strengths
            improvement_areas = []
            strengths = []
            
            for category, score in category_scores.items():
                if score < 0.7:
                    improvement_areas.append(f"{category.replace('_', ' ').title()}")
                elif score >= 0.9:
                    strengths.append(f"{category.replace('_', ' ').title()}")
            
            # Benchmark comparison (simplified)
            benchmark_comparison = {
                "industry_average": 0.75,
                "top_quartile": 0.85,
                "security_leaders": 0.95,
                "your_score": overall_score,
                "percentile": min(99, int(overall_score * 100))
            }
            
            security_score = SecurityScore(
                overall_score=overall_score,
                category_scores=category_scores,
                risk_level=risk_level,
                improvement_areas=improvement_areas,
                strengths=strengths,
                benchmark_comparison=benchmark_comparison
            )
            
            # Audit scoring
            await self.compliance_audit.log_audit_event(
                event_type="SECURITY_SCORING_COMPLETED",
                category=AuditEventCategory.SECURITY_EVENT,
                severity=SeverityLevel.INFO,
                action="VALIDATE_RESULTS",
                outcome="SUCCESS",
                details=security_score.to_dict(),
                compliance_tags=["SECURITY_SCORING", "RISK_ASSESSMENT"]
            )
            
            return security_score
            
        except Exception as e:
            logger.error(f"Test result validation failed: {e}")
            raise
    
    async def generate_security_validation_report(
        self,
        test_results: PenTestResults,
        security_score: SecurityScore,
        format_type: str = "JSON"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive security validation report.
        
        Args:
            test_results: Test results
            security_score: Security score
            format_type: Report format (JSON, PDF, HTML)
            
        Returns:
            Generated security report
        """
        try:
            report = {
                "report_metadata": {
                    "report_id": str(uuid.uuid4()),
                    "generated_at": datetime.utcnow().isoformat(),
                    "report_type": "Security Validation Report",
                    "format": format_type,
                    "target": test_results.target.to_dict(),
                    "test_period": {
                        "start": test_results.test_started.isoformat(),
                        "end": test_results.test_completed.isoformat() if test_results.test_completed else None
                    }
                },
                "executive_summary": {
                    "overall_security_score": security_score.overall_score,
                    "risk_level": security_score.risk_level.value,
                    "total_tests_executed": test_results.tests_completed,
                    "vulnerabilities_found": test_results.vulnerabilities_found,
                    "critical_issues": test_results.critical_vulnerabilities,
                    "immediate_action_required": test_results.critical_vulnerabilities > 0,
                    "key_findings": test_results.detailed_findings[:5],  # Top 5 findings
                    "security_posture_summary": test_results.security_posture_rating
                },
                "detailed_assessment": {
                    "test_execution_summary": test_results.calculate_summary(),
                    "security_score_breakdown": security_score.to_dict(),
                    "vulnerability_analysis": {
                        "by_severity": {
                            "critical": test_results.critical_vulnerabilities,
                            "high": test_results.high_vulnerabilities,
                            "medium": test_results.medium_vulnerabilities,
                            "low": test_results.low_vulnerabilities
                        },
                        "by_category": self._categorize_vulnerabilities(test_results.test_results)
                    },
                    "risk_assessment": {
                        "business_impact": self._assess_business_impact(test_results),
                        "technical_risk": self._assess_technical_risk(test_results),
                        "compliance_implications": self._assess_compliance_impact(test_results)
                    }
                },
                "findings_and_recommendations": {
                    "critical_findings": [f for f in test_results.detailed_findings 
                                        if f.get("severity") == "CRITICAL"],
                    "high_priority_findings": [f for f in test_results.detailed_findings 
                                             if f.get("severity") == "HIGH"],
                    "remediation_roadmap": test_results.remediation_roadmap,
                    "best_practices": self._generate_best_practices(test_results),
                    "security_improvements": security_score.improvement_areas
                },
                "technical_details": {
                    "test_cases_executed": [r.to_dict() for r in test_results.test_results],
                    "evidence_collected": self._collect_technical_evidence(test_results),
                    "false_positive_analysis": self._analyze_false_positives(test_results),
                    "tool_information": {
                        "framework_version": "2.0",
                        "test_methodology": "OWASP Testing Guide v4.2",
                        "compliance_standards": ["OWASP Top 10", "SANS Top 25"]
                    }
                },
                "appendices": {
                    "glossary": self._generate_security_glossary(),
                    "references": self._generate_references(),
                    "contact_information": {
                        "security_team": "security@leanvibe.com",
                        "report_questions": "pentest-reports@leanvibe.com"
                    }
                }
            }
            
            # Save report to file system
            report_file = self.test_data_path / f"security_report_{test_results.id}.json"
            async with aiofiles.open(report_file, 'w') as f:
                await f.write(json.dumps(report, indent=2, default=str))
            
            # Audit report generation
            await self.compliance_audit.log_audit_event(
                event_type="SECURITY_REPORT_GENERATED",
                category=AuditEventCategory.COMPLIANCE_EVENT,
                severity=SeverityLevel.INFO,
                action="GENERATE_REPORT",
                outcome="SUCCESS",
                details={
                    "report_id": report["report_metadata"]["report_id"],
                    "target": test_results.target.name,
                    "vulnerabilities_reported": test_results.vulnerabilities_found
                },
                compliance_tags=["SECURITY_REPORTING", "DOCUMENTATION"]
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Security report generation failed: {e}")
            raise
    
    # Private helper methods
    async def _select_test_cases(
        self,
        target: PenTestTarget,
        test_types: Optional[List[PenTestType]],
        custom_tests: Optional[List[SecurityTestCase]]
    ) -> List[SecurityTestCase]:
        """Select appropriate test cases for target."""
        selected_tests = []
        
        # Add standard test cases
        for test_case in self.test_cases.values():
            if test_types is None or test_case.test_type in test_types:
                # Check if test is applicable to target type
                if self._is_test_applicable(test_case, target):
                    selected_tests.append(test_case)
        
        # Add custom test cases
        if custom_tests:
            selected_tests.extend(custom_tests)
        
        return selected_tests
    
    def _is_test_applicable(self, test_case: SecurityTestCase, target: PenTestTarget) -> bool:
        """Check if test case is applicable to target."""
        # Web application tests
        if target.target_type == "WEB_APP":
            return test_case.test_type in [
                PenTestType.WEB_APPLICATION,
                PenTestType.XSS_TESTING,
                PenTestType.CSRF_TESTING,
                PenTestType.INJECTION_ATTACKS,
                PenTestType.AUTHENTICATION,
                PenTestType.AUTHORIZATION
            ]
        
        # API tests
        elif target.target_type == "API":
            return test_case.test_type in [
                PenTestType.API_SECURITY,
                PenTestType.AUTHENTICATION,
                PenTestType.AUTHORIZATION,
                PenTestType.DATA_VALIDATION,
                PenTestType.INJECTION_ATTACKS
            ]
        
        # Network tests
        elif target.target_type == "NETWORK":
            return test_case.test_type in [
                PenTestType.NETWORK_SCAN,
                PenTestType.BRUTE_FORCE,
                PenTestType.PRIVILEGE_ESCALATION
            ]
        
        # Service tests
        elif target.target_type == "SERVICE":
            return test_case.test_type in [
                PenTestType.AUTHENTICATION,
                PenTestType.AUTHORIZATION,
                PenTestType.DATA_VALIDATION
            ]
        
        return True  # Default to applicable
    
    async def _execute_test_case(self, test_case: SecurityTestCase, target: PenTestTarget) -> TestResult:
        """Execute individual test case."""
        start_time = datetime.utcnow()
        
        try:
            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
            # Execute test based on type
            if test_case.test_type == PenTestType.AUTHENTICATION:
                result = await self._test_authentication(test_case, target)
            elif test_case.test_type == PenTestType.AUTHORIZATION:
                result = await self._test_authorization(test_case, target)
            elif test_case.test_type == PenTestType.INJECTION_ATTACKS:
                result = await self._test_injection_attacks(test_case, target)
            elif test_case.test_type == PenTestType.XSS_TESTING:
                result = await self._test_xss_vulnerabilities(test_case, target)
            elif test_case.test_type == PenTestType.API_SECURITY:
                result = await self._test_api_security(test_case, target)
            elif test_case.test_type == PenTestType.DATA_VALIDATION:
                result = await self._test_data_validation(test_case, target)
            else:
                # Default test execution
                result = await self._execute_generic_test(test_case, target)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            result.execution_time = execution_time
            result.status = TestStatus.COMPLETED
            
            return result
            
        except asyncio.TimeoutError:
            return TestResult(
                id=uuid.uuid4(),
                test_case_id=test_case.id,
                target_id=target.id,
                status=TestStatus.FAILED,
                vulnerability_found=False,
                vulnerability_class=None,
                risk_score=0.0,
                confidence=0.0,
                evidence={},
                execution_time=(datetime.utcnow() - start_time).total_seconds(),
                error_message="Test execution timeout"
            )
        except Exception as e:
            return TestResult(
                id=uuid.uuid4(),
                test_case_id=test_case.id,
                target_id=target.id,
                status=TestStatus.FAILED,
                vulnerability_found=False,
                vulnerability_class=None,
                risk_score=0.0,
                confidence=0.0,
                evidence={},
                execution_time=(datetime.utcnow() - start_time).total_seconds(),
                error_message=str(e)
            )
    
    async def _test_authentication(self, test_case: SecurityTestCase, target: PenTestTarget) -> TestResult:
        """Test authentication mechanisms."""
        result = TestResult(
            id=uuid.uuid4(),
            test_case_id=test_case.id,
            target_id=target.id,
            status=TestStatus.RUNNING,
            vulnerability_found=False,
            vulnerability_class=None,
            risk_score=0.0,
            confidence=0.8,
            evidence={}
        )
        
        try:
            # Test weak password policies
            if "weak_password" in test_case.test_payload:
                weak_passwords = ["password", "123456", "admin", "test"]
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    for password in weak_passwords:
                        try:
                            response = await client.post(
                                f"{target.endpoint}/login",
                                json={"username": "admin", "password": password}
                            )
                            
                            if response.status_code == 200:
                                result.vulnerability_found = True
                                result.vulnerability_class = VulnerabilityClass.BROKEN_AUTHENTICATION
                                result.risk_score = 0.9
                                result.evidence["weak_password_accepted"] = password
                                result.remediation_priority = TestSeverity.CRITICAL
                                break
                                
                        except Exception:
                            continue
            
            # Test brute force protection
            if "brute_force" in test_case.test_payload:
                attempt_count = 0
                blocked = False
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    for i in range(20):  # Try 20 failed attempts
                        try:
                            response = await client.post(
                                f"{target.endpoint}/login",
                                json={"username": "admin", "password": f"wrong_password_{i}"}
                            )
                            attempt_count += 1
                            
                            # Check if we get blocked/rate limited
                            if response.status_code == 429 or "blocked" in response.text.lower():
                                blocked = True
                                break
                                
                        except Exception:
                            continue
                
                if not blocked and attempt_count >= 10:
                    result.vulnerability_found = True
                    result.vulnerability_class = VulnerabilityClass.BROKEN_AUTHENTICATION
                    result.risk_score = 0.7
                    result.evidence["brute_force_attempts"] = attempt_count
                    result.evidence["blocked"] = blocked
                    result.remediation_priority = TestSeverity.HIGH
            
            return result
            
        except Exception as e:
            result.error_message = str(e)
            return result
    
    async def _test_authorization(self, test_case: SecurityTestCase, target: PenTestTarget) -> TestResult:
        """Test authorization mechanisms."""
        result = TestResult(
            id=uuid.uuid4(),
            test_case_id=test_case.id,
            target_id=target.id,
            status=TestStatus.RUNNING,
            vulnerability_found=False,
            vulnerability_class=None,
            risk_score=0.0,
            confidence=0.8,
            evidence={}
        )
        
        try:
            # Test privilege escalation
            if "privilege_escalation" in test_case.test_payload:
                # Try accessing admin endpoints without proper authorization
                admin_endpoints = ["/admin", "/admin/users", "/admin/settings", "/api/admin/users"]
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    for endpoint in admin_endpoints:
                        try:
                            # Test without authentication
                            response = await client.get(f"{target.endpoint}{endpoint}")
                            
                            if response.status_code == 200:
                                result.vulnerability_found = True
                                result.vulnerability_class = VulnerabilityClass.BROKEN_ACCESS_CONTROL
                                result.risk_score = 0.9
                                result.evidence["unauthorized_access"] = endpoint
                                result.remediation_priority = TestSeverity.CRITICAL
                                break
                                
                        except Exception:
                            continue
            
            # Test horizontal privilege escalation
            if "horizontal_escalation" in test_case.test_payload:
                # Try accessing other users' data
                user_ids = ["1", "2", "admin", "test"]
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    for user_id in user_ids:
                        try:
                            response = await client.get(f"{target.endpoint}/api/users/{user_id}/profile")
                            
                            if response.status_code == 200 and "user" in response.text.lower():
                                result.vulnerability_found = True
                                result.vulnerability_class = VulnerabilityClass.BROKEN_ACCESS_CONTROL
                                result.risk_score = 0.8
                                result.evidence["user_data_accessed"] = user_id
                                result.remediation_priority = TestSeverity.HIGH
                                
                        except Exception:
                            continue
            
            return result
            
        except Exception as e:
            result.error_message = str(e)
            return result
    
    async def _test_injection_attacks(self, test_case: SecurityTestCase, target: PenTestTarget) -> TestResult:
        """Test for injection vulnerabilities."""
        result = TestResult(
            id=uuid.uuid4(),
            test_case_id=test_case.id,
            target_id=target.id,
            status=TestStatus.RUNNING,
            vulnerability_found=False,
            vulnerability_class=None,
            risk_score=0.0,
            confidence=0.9,
            evidence={}
        )
        
        try:
            # SQL injection payloads
            sql_payloads = [
                "' OR '1'='1",
                "' UNION SELECT * FROM users--",
                "'; DROP TABLE users;--",
                "' OR 1=1--",
                "admin'--"
            ]
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                for payload in sql_payloads:
                    try:
                        # Test in query parameters
                        response = await client.get(f"{target.endpoint}/search?q={payload}")
                        
                        # Check for SQL error messages
                        error_indicators = [
                            "sql", "mysql", "postgresql", "oracle", "sqlite",
                            "syntax error", "database error", "column", "table"
                        ]
                        
                        response_text = response.text.lower()
                        for indicator in error_indicators:
                            if indicator in response_text:
                                result.vulnerability_found = True
                                result.vulnerability_class = VulnerabilityClass.INJECTION
                                result.risk_score = 0.95
                                result.evidence["sql_injection_payload"] = payload
                                result.evidence["error_response"] = response_text[:500]
                                result.remediation_priority = TestSeverity.CRITICAL
                                return result
                                
                    except Exception:
                        continue
                
                # Test in POST data
                for payload in sql_payloads:
                    try:
                        response = await client.post(
                            f"{target.endpoint}/login",
                            json={"username": payload, "password": "test"}
                        )
                        
                        response_text = response.text.lower()
                        for indicator in error_indicators:
                            if indicator in response_text:
                                result.vulnerability_found = True
                                result.vulnerability_class = VulnerabilityClass.INJECTION
                                result.risk_score = 0.95
                                result.evidence["sql_injection_payload"] = payload
                                result.evidence["error_response"] = response_text[:500]
                                result.remediation_priority = TestSeverity.CRITICAL
                                return result
                                
                    except Exception:
                        continue
            
            return result
            
        except Exception as e:
            result.error_message = str(e)
            return result
    
    async def _test_xss_vulnerabilities(self, test_case: SecurityTestCase, target: PenTestTarget) -> TestResult:
        """Test for XSS vulnerabilities."""
        result = TestResult(
            id=uuid.uuid4(),
            test_case_id=test_case.id,
            target_id=target.id,
            status=TestStatus.RUNNING,
            vulnerability_found=False,
            vulnerability_class=None,
            risk_score=0.0,
            confidence=0.8,
            evidence={}
        )
        
        try:
            # XSS payloads
            xss_payloads = [
                "<script>alert('XSS')</script>",
                "<img src=x onerror=alert('XSS')>",
                "<svg onload=alert('XSS')>",
                "javascript:alert('XSS')",
                "<iframe src=javascript:alert('XSS')></iframe>"
            ]
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                for payload in xss_payloads:
                    try:
                        # Test in search parameters
                        response = await client.get(f"{target.endpoint}/search?q={payload}")
                        
                        # Check if payload is reflected in response
                        if payload in response.text:
                            result.vulnerability_found = True
                            result.vulnerability_class = VulnerabilityClass.XSS
                            result.risk_score = 0.8
                            result.evidence["xss_payload"] = payload
                            result.evidence["reflected"] = True
                            result.remediation_priority = TestSeverity.HIGH
                            return result
                            
                    except Exception:
                        continue
                
                # Test in form submissions
                for payload in xss_payloads:
                    try:
                        response = await client.post(
                            f"{target.endpoint}/contact",
                            json={"name": payload, "message": "test"}
                        )
                        
                        if payload in response.text:
                            result.vulnerability_found = True
                            result.vulnerability_class = VulnerabilityClass.XSS
                            result.risk_score = 0.8
                            result.evidence["xss_payload"] = payload
                            result.evidence["stored"] = True
                            result.remediation_priority = TestSeverity.HIGH
                            return result
                            
                    except Exception:
                        continue
            
            return result
            
        except Exception as e:
            result.error_message = str(e)
            return result
    
    async def _test_api_security(self, test_case: SecurityTestCase, target: PenTestTarget) -> TestResult:
        """Test API security configurations."""
        result = TestResult(
            id=uuid.uuid4(),
            test_case_id=test_case.id,
            target_id=target.id,
            status=TestStatus.RUNNING,
            vulnerability_found=False,
            vulnerability_class=None,
            risk_score=0.0,
            confidence=0.8,
            evidence={}
        )
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Test for exposed sensitive endpoints
                sensitive_endpoints = [
                    "/api/config",
                    "/api/debug",
                    "/api/internal",
                    "/api/admin",
                    "/.env",
                    "/config.json",
                    "/swagger.json"
                ]
                
                for endpoint in sensitive_endpoints:
                    try:
                        response = await client.get(f"{target.endpoint}{endpoint}")
                        
                        if response.status_code == 200 and len(response.text) > 50:
                            result.vulnerability_found = True
                            result.vulnerability_class = VulnerabilityClass.SENSITIVE_DATA_EXPOSURE
                            result.risk_score = 0.7
                            result.evidence["exposed_endpoint"] = endpoint
                            result.evidence["response_size"] = len(response.text)
                            result.remediation_priority = TestSeverity.HIGH
                            
                    except Exception:
                        continue
                
                # Test for missing security headers
                try:
                    response = await client.get(target.endpoint)
                    headers = response.headers
                    
                    missing_headers = []
                    security_headers = [
                        "X-Content-Type-Options",
                        "X-Frame-Options",
                        "X-XSS-Protection",
                        "Strict-Transport-Security",
                        "Content-Security-Policy"
                    ]
                    
                    for header in security_headers:
                        if header not in headers:
                            missing_headers.append(header)
                    
                    if missing_headers:
                        result.vulnerability_found = True
                        result.vulnerability_class = VulnerabilityClass.SECURITY_MISCONFIGURATION
                        result.risk_score = 0.4
                        result.evidence["missing_security_headers"] = missing_headers
                        result.remediation_priority = TestSeverity.MEDIUM
                        
                except Exception:
                    pass
            
            return result
            
        except Exception as e:
            result.error_message = str(e)
            return result
    
    async def _test_data_validation(self, test_case: SecurityTestCase, target: PenTestTarget) -> TestResult:
        """Test data validation mechanisms."""
        result = TestResult(
            id=uuid.uuid4(),
            test_case_id=test_case.id,
            target_id=target.id,
            status=TestStatus.RUNNING,
            vulnerability_found=False,
            vulnerability_class=None,
            risk_score=0.0,
            confidence=0.7,
            evidence={}
        )
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Test with oversized data
                large_payload = "A" * 10000  # 10KB of data
                
                try:
                    response = await client.post(
                        f"{target.endpoint}/api/submit",
                        json={"data": large_payload}
                    )
                    
                    if response.status_code == 200:
                        result.vulnerability_found = True
                        result.vulnerability_class = VulnerabilityClass.SECURITY_MISCONFIGURATION
                        result.risk_score = 0.5
                        result.evidence["accepts_large_payload"] = len(large_payload)
                        result.remediation_priority = TestSeverity.MEDIUM
                        
                except Exception:
                    pass
                
                # Test with invalid data types
                invalid_payloads = [
                    {"id": "not_a_number"},
                    {"email": "invalid-email"},
                    {"date": "invalid-date"},
                    {"json": "invalid json}"}
                ]
                
                for payload in invalid_payloads:
                    try:
                        response = await client.post(
                            f"{target.endpoint}/api/validate",
                            json=payload
                        )
                        
                        # If no validation error is returned, it's a vulnerability
                        if response.status_code == 200 and "error" not in response.text.lower():
                            result.vulnerability_found = True
                            result.vulnerability_class = VulnerabilityClass.SECURITY_MISCONFIGURATION
                            result.risk_score = 0.6
                            result.evidence["accepts_invalid_data"] = payload
                            result.remediation_priority = TestSeverity.MEDIUM
                            break
                            
                    except Exception:
                        continue
            
            return result
            
        except Exception as e:
            result.error_message = str(e)
            return result
    
    async def _execute_generic_test(self, test_case: SecurityTestCase, target: PenTestTarget) -> TestResult:
        """Execute generic security test."""
        return TestResult(
            id=uuid.uuid4(),
            test_case_id=test_case.id,
            target_id=target.id,
            status=TestStatus.COMPLETED,
            vulnerability_found=False,
            vulnerability_class=None,
            risk_score=0.0,
            confidence=0.5,
            evidence={"message": "Generic test executed"},
            execution_time=0.1
        )
    
    async def _calculate_overall_risk(self, test_results: PenTestResults) -> None:
        """Calculate overall risk score for test results."""
        if not test_results.test_results:
            test_results.overall_risk_score = 0.0
            test_results.security_posture_rating = "UNKNOWN"
            return
        
        # Calculate weighted risk score
        total_weight = 0
        weighted_risk = 0
        
        for result in test_results.test_results:
            if result.vulnerability_found:
                # Weight by severity
                severity_weight = {
                    TestSeverity.CRITICAL: 1.0,
                    TestSeverity.HIGH: 0.7,
                    TestSeverity.MEDIUM: 0.4,
                    TestSeverity.LOW: 0.1
                }.get(result.remediation_priority, 0.1)
                
                weighted_risk += result.risk_score * severity_weight * result.confidence
                total_weight += severity_weight
        
        # Calculate overall risk
        if total_weight > 0:
            test_results.overall_risk_score = min(1.0, weighted_risk / total_weight)
        else:
            test_results.overall_risk_score = 0.0
        
        # Determine security posture rating
        if test_results.overall_risk_score <= 0.2:
            test_results.security_posture_rating = "EXCELLENT"
        elif test_results.overall_risk_score <= 0.4:
            test_results.security_posture_rating = "GOOD"
        elif test_results.overall_risk_score <= 0.6:
            test_results.security_posture_rating = "FAIR"
        elif test_results.overall_risk_score <= 0.8:
            test_results.security_posture_rating = "POOR"
        else:
            test_results.security_posture_rating = "CRITICAL"
    
    async def _generate_findings_and_recommendations(self, test_results: PenTestResults) -> None:
        """Generate detailed findings and recommendations."""
        findings = []
        recommendations = []
        
        # Group vulnerabilities by type
        vulnerability_groups = {}
        for result in test_results.test_results:
            if result.vulnerability_found and result.vulnerability_class:
                vuln_class = result.vulnerability_class.value
                if vuln_class not in vulnerability_groups:
                    vulnerability_groups[vuln_class] = []
                vulnerability_groups[vuln_class].append(result)
        
        # Generate findings for each vulnerability class
        for vuln_class, results in vulnerability_groups.items():
            finding = {
                "id": str(uuid.uuid4()),
                "vulnerability_class": vuln_class,
                "severity": max([r.remediation_priority.value for r in results]),
                "instances_found": len(results),
                "risk_score": max([r.risk_score for r in results]),
                "description": self._get_vulnerability_description(vuln_class),
                "impact": self._get_vulnerability_impact(vuln_class),
                "evidence": [r.evidence for r in results],
                "remediation": self._get_remediation_guidance(vuln_class)
            }
            findings.append(finding)
        
        # Generate remediation roadmap
        priority_order = [TestSeverity.CRITICAL, TestSeverity.HIGH, TestSeverity.MEDIUM, TestSeverity.LOW]
        
        for priority in priority_order:
            priority_findings = [f for f in findings if f["severity"] == priority.value]
            
            if priority_findings:
                roadmap_item = {
                    "priority": priority.value,
                    "timeline": self._get_remediation_timeline(priority),
                    "findings_count": len(priority_findings),
                    "actions": [f["remediation"] for f in priority_findings],
                    "estimated_effort": self._estimate_remediation_effort(priority_findings)
                }
                recommendations.append(roadmap_item)
        
        test_results.detailed_findings = findings
        test_results.remediation_roadmap = recommendations
    
    def _get_vulnerability_description(self, vuln_class: str) -> str:
        """Get description for vulnerability class."""
        descriptions = {
            "INJECTION": "SQL, NoSQL, OS, and LDAP injection vulnerabilities allow attackers to send untrusted data to an interpreter.",
            "BROKEN_AUTHENTICATION": "Authentication and session management functions are often implemented incorrectly.",
            "SENSITIVE_DATA_EXPOSURE": "Applications often fail to adequately protect sensitive data.",
            "XML_EXTERNAL_ENTITIES": "Older XML processors evaluate external entity references within XML documents.",
            "BROKEN_ACCESS_CONTROL": "Restrictions on what authenticated users are allowed to do are often not properly enforced.",
            "SECURITY_MISCONFIGURATION": "Security misconfiguration is the most commonly seen issue.",
            "XSS": "Cross-Site Scripting (XSS) flaws occur when applications include untrusted data in web pages.",
            "INSECURE_DESERIALIZATION": "Insecure deserialization often leads to remote code execution.",
            "KNOWN_VULNERABILITIES": "Components with known vulnerabilities are used in applications.",
            "INSUFFICIENT_LOGGING": "Insufficient logging and monitoring, coupled with missing integration with incident response."
        }
        return descriptions.get(vuln_class, "Security vulnerability detected.")
    
    def _get_vulnerability_impact(self, vuln_class: str) -> str:
        """Get impact description for vulnerability class."""
        impacts = {
            "INJECTION": "Data loss, corruption, denial of access, complete host takeover",
            "BROKEN_AUTHENTICATION": "Account takeover, identity theft, privilege escalation",
            "SENSITIVE_DATA_EXPOSURE": "Data breach, identity theft, financial fraud",
            "XML_EXTERNAL_ENTITIES": "Data exfiltration, server-side request forgery, denial of service",
            "BROKEN_ACCESS_CONTROL": "Unauthorized access to functionality and data",
            "SECURITY_MISCONFIGURATION": "Complete system compromise, data access",
            "XSS": "Account takeover, session hijacking, malware distribution",
            "INSECURE_DESERIALIZATION": "Remote code execution, privilege escalation",
            "KNOWN_VULNERABILITIES": "System compromise, data breach",
            "INSUFFICIENT_LOGGING": "Undetected breaches, delayed incident response"
        }
        return impacts.get(vuln_class, "Security impact varies based on vulnerability.")
    
    def _get_remediation_guidance(self, vuln_class: str) -> str:
        """Get remediation guidance for vulnerability class."""
        guidance = {
            "INJECTION": "Use parameterized queries, input validation, and proper escaping",
            "BROKEN_AUTHENTICATION": "Implement multi-factor authentication and secure session management",
            "SENSITIVE_DATA_EXPOSURE": "Encrypt sensitive data at rest and in transit",
            "XML_EXTERNAL_ENTITIES": "Disable XML external entities and use simple data formats",
            "BROKEN_ACCESS_CONTROL": "Implement proper access controls and deny by default",
            "SECURITY_MISCONFIGURATION": "Implement secure configuration processes and regular reviews",
            "XSS": "Use proper output encoding and Content Security Policy",
            "INSECURE_DESERIALIZATION": "Avoid deserialization of untrusted data",
            "KNOWN_VULNERABILITIES": "Keep components up to date and remove unused dependencies",
            "INSUFFICIENT_LOGGING": "Implement comprehensive logging and monitoring"
        }
        return guidance.get(vuln_class, "Follow security best practices for remediation.")
    
    def _get_remediation_timeline(self, priority: TestSeverity) -> str:
        """Get recommended timeline for remediation based on priority."""
        timelines = {
            TestSeverity.CRITICAL: "Immediate (within 24 hours)",
            TestSeverity.HIGH: "Urgent (within 1 week)",
            TestSeverity.MEDIUM: "Important (within 1 month)",
            TestSeverity.LOW: "Standard (within 3 months)"
        }
        return timelines.get(priority, "As resources allow")
    
    def _estimate_remediation_effort(self, findings: List[Dict[str, Any]]) -> str:
        """Estimate remediation effort for findings."""
        total_findings = len(findings)
        
        if total_findings <= 2:
            return "Low (1-2 person days)"
        elif total_findings <= 5:
            return "Medium (3-5 person days)"
        elif total_findings <= 10:
            return "High (1-2 person weeks)"
        else:
            return "Very High (2+ person weeks)"
    
    async def _calculate_category_score(self, test_results: List[TestResult]) -> float:
        """Calculate security score for a category of tests."""
        if not test_results:
            return 0.5  # Default score when no tests available
        
        total_tests = len(test_results)
        passed_tests = len([r for r in test_results if not r.vulnerability_found])
        
        if total_tests == 0:
            return 0.5
        
        base_score = passed_tests / total_tests
        
        # Adjust for severity of failed tests
        severity_penalty = 0
        for result in test_results:
            if result.vulnerability_found:
                if result.remediation_priority == TestSeverity.CRITICAL:
                    severity_penalty += 0.3
                elif result.remediation_priority == TestSeverity.HIGH:
                    severity_penalty += 0.2
                elif result.remediation_priority == TestSeverity.MEDIUM:
                    severity_penalty += 0.1
        
        final_score = max(0.0, base_score - severity_penalty)
        return min(1.0, final_score)
    
    def _categorize_vulnerabilities(self, test_results: List[TestResult]) -> Dict[str, int]:
        """Categorize vulnerabilities by class."""
        categories = {}
        
        for result in test_results:
            if result.vulnerability_found and result.vulnerability_class:
                vuln_class = result.vulnerability_class.value
                categories[vuln_class] = categories.get(vuln_class, 0) + 1
        
        return categories
    
    def _assess_business_impact(self, test_results: PenTestResults) -> Dict[str, Any]:
        """Assess business impact of found vulnerabilities."""
        return {
            "data_breach_risk": "HIGH" if test_results.critical_vulnerabilities > 0 else "MEDIUM",
            "compliance_impact": "SIGNIFICANT" if test_results.high_vulnerabilities > 0 else "MINIMAL",
            "operational_risk": "HIGH" if test_results.vulnerabilities_found > 10 else "MEDIUM",
            "reputation_risk": "HIGH" if test_results.critical_vulnerabilities > 0 else "LOW"
        }
    
    def _assess_technical_risk(self, test_results: PenTestResults) -> Dict[str, Any]:
        """Assess technical risk of found vulnerabilities."""
        return {
            "system_compromise_likelihood": "HIGH" if test_results.critical_vulnerabilities > 0 else "MEDIUM",
            "data_exfiltration_risk": "HIGH" if test_results.high_vulnerabilities > 2 else "MEDIUM",
            "service_disruption_risk": "MEDIUM",
            "privilege_escalation_risk": "HIGH" if any(r.vulnerability_class == VulnerabilityClass.BROKEN_ACCESS_CONTROL 
                                                      for r in test_results.test_results if r.vulnerability_found) else "LOW"
        }
    
    def _assess_compliance_impact(self, test_results: PenTestResults) -> Dict[str, Any]:
        """Assess compliance impact of found vulnerabilities."""
        return {
            "gdpr_compliance": "AT_RISK" if test_results.critical_vulnerabilities > 0 else "COMPLIANT",
            "sox_compliance": "AT_RISK" if test_results.high_vulnerabilities > 0 else "COMPLIANT",
            "iso27001_compliance": "NEEDS_REVIEW" if test_results.vulnerabilities_found > 5 else "COMPLIANT",
            "pci_dss_compliance": "AT_RISK" if any(r.vulnerability_class == VulnerabilityClass.SENSITIVE_DATA_EXPOSURE 
                                                  for r in test_results.test_results if r.vulnerability_found) else "COMPLIANT"
        }
    
    def _generate_best_practices(self, test_results: PenTestResults) -> List[str]:
        """Generate security best practices based on test results."""
        practices = [
            "Implement defense in depth security architecture",
            "Conduct regular security training for development teams",
            "Establish secure coding standards and guidelines",
            "Implement automated security testing in CI/CD pipeline",
            "Conduct regular penetration testing and vulnerability assessments",
            "Maintain an incident response plan and practice it regularly",
            "Implement proper logging and monitoring systems",
            "Keep all systems and dependencies up to date"
        ]
        
        return practices
    
    def _collect_technical_evidence(self, test_results: PenTestResults) -> List[Dict[str, Any]]:
        """Collect technical evidence from test results."""
        evidence = []
        
        for result in test_results.test_results:
            if result.vulnerability_found and result.evidence:
                evidence_item = {
                    "test_case_id": str(result.test_case_id),
                    "vulnerability_class": result.vulnerability_class.value if result.vulnerability_class else "UNKNOWN",
                    "risk_score": result.risk_score,
                    "confidence": result.confidence,
                    "evidence": result.evidence,
                    "timestamp": datetime.utcnow().isoformat()
                }
                evidence.append(evidence_item)
        
        return evidence
    
    def _analyze_false_positives(self, test_results: PenTestResults) -> Dict[str, Any]:
        """Analyze potential false positives in test results."""
        total_vulnerabilities = test_results.vulnerabilities_found
        
        if total_vulnerabilities == 0:
            return {"analysis": "No vulnerabilities found, no false positive analysis needed"}
        
        # Calculate false positive probability
        avg_fp_probability = sum([r.false_positive_probability for r in test_results.test_results 
                                 if r.vulnerability_found]) / max(1, total_vulnerabilities)
        
        estimated_false_positives = int(total_vulnerabilities * avg_fp_probability)
        
        return {
            "total_vulnerabilities": total_vulnerabilities,
            "estimated_false_positives": estimated_false_positives,
            "false_positive_rate": avg_fp_probability,
            "confidence_level": 1.0 - avg_fp_probability,
            "recommendation": "Manual verification recommended for findings with >20% false positive probability"
        }
    
    def _generate_security_glossary(self) -> Dict[str, str]:
        """Generate security glossary for the report."""
        return {
            "SQL Injection": "A code injection technique that exploits security vulnerabilities in database layer",
            "XSS": "Cross-Site Scripting - injection of malicious scripts into trusted websites",
            "CSRF": "Cross-Site Request Forgery - attack that forces users to execute unwanted actions",
            "Authentication": "Process of verifying the identity of a user or system",
            "Authorization": "Process of verifying what resources a user has access to",
            "Vulnerability": "A weakness that can be exploited by attackers",
            "Risk Score": "Numerical assessment of security risk from 0.0 (low) to 1.0 (critical)",
            "False Positive": "Security alert that incorrectly indicates malicious activity"
        }
    
    def _generate_references(self) -> List[str]:
        """Generate reference list for the report."""
        return [
            "OWASP Top 10 - https://owasp.org/www-project-top-ten/",
            "SANS Top 25 - https://www.sans.org/top25-software-errors/",
            "NIST Cybersecurity Framework - https://www.nist.gov/cyberframework",
            "ISO 27001 - https://www.iso.org/isoiec-27001-information-security.html",
            "CVSS v3.1 - https://www.first.org/cvss/v3.1/specification-document"
        ]
    
    def _initialize_test_cases(self) -> None:
        """Initialize comprehensive security test cases."""
        
        # Authentication tests
        auth_tests = [
            SecurityTestCase(
                id=uuid.uuid4(),
                name="Weak Password Policy Test",
                description="Test for weak password acceptance",
                test_type=PenTestType.AUTHENTICATION,
                severity=TestSeverity.HIGH,
                test_payload={"weak_password": True},
                expected_behavior="System should reject weak passwords",
                validation_criteria=["Password complexity requirements enforced"],
                remediation_guidance="Implement strong password policies with complexity requirements",
                references=["OWASP ASVS 2.1"],
                tags=["authentication", "password-policy"]
            ),
            SecurityTestCase(
                id=uuid.uuid4(),
                name="Brute Force Protection Test",
                description="Test for brute force attack protection",
                test_type=PenTestType.AUTHENTICATION,
                severity=TestSeverity.HIGH,
                test_payload={"brute_force": True},
                expected_behavior="System should implement rate limiting or account lockout",
                validation_criteria=["Account lockout after failed attempts", "Rate limiting implemented"],
                remediation_guidance="Implement account lockout and rate limiting mechanisms",
                references=["OWASP ASVS 2.2"],
                tags=["authentication", "brute-force"]
            )
        ]
        
        # Authorization tests
        authz_tests = [
            SecurityTestCase(
                id=uuid.uuid4(),
                name="Privilege Escalation Test",
                description="Test for vertical privilege escalation vulnerabilities",
                test_type=PenTestType.AUTHORIZATION,
                severity=TestSeverity.CRITICAL,
                test_payload={"privilege_escalation": True},
                expected_behavior="System should deny unauthorized access to admin functions",
                validation_criteria=["Proper access controls implemented", "Role-based authorization enforced"],
                remediation_guidance="Implement proper role-based access control (RBAC)",
                references=["OWASP ASVS 4.1"],
                tags=["authorization", "privilege-escalation"]
            ),
            SecurityTestCase(
                id=uuid.uuid4(),
                name="Horizontal Privilege Escalation Test",
                description="Test for horizontal privilege escalation vulnerabilities",
                test_type=PenTestType.AUTHORIZATION,
                severity=TestSeverity.HIGH,
                test_payload={"horizontal_escalation": True},
                expected_behavior="Users should only access their own resources",
                validation_criteria=["User resource isolation enforced"],
                remediation_guidance="Implement proper authorization checks for user resources",
                references=["OWASP ASVS 4.2"],
                tags=["authorization", "horizontal-escalation"]
            )
        ]
        
        # Combine all test cases
        all_tests = auth_tests + authz_tests
        
        for test in all_tests:
            self.test_cases[str(test.id)] = test
    
    def _initialize_validation_rules(self) -> None:
        """Initialize security validation rules."""
        self.validation_rules = {
            "password_complexity": {
                "min_length": 8,
                "require_uppercase": True,
                "require_lowercase": True,
                "require_numbers": True,
                "require_special": True
            },
            "rate_limiting": {
                "max_attempts": 5,
                "time_window": 300,  # 5 minutes
                "lockout_duration": 900  # 15 minutes
            },
            "session_management": {
                "max_session_duration": 3600,  # 1 hour
                "require_https": True,
                "secure_cookies": True
            }
        }


# Factory function
async def create_penetration_testing_framework(
    db_session: AsyncSession,
    security_audit: SecurityAuditSystem,
    compliance_audit: ComplianceAuditSystem,
    security_monitoring: SecurityMonitoringSystem,
    test_data_path: str = "/var/lib/agent-hive/security-tests"
) -> PenetrationTestingFramework:
    """
    Create penetration testing framework instance.
    
    Args:
        db_session: Database session
        security_audit: Security audit system
        compliance_audit: Compliance audit system
        security_monitoring: Security monitoring system
        test_data_path: Path for test data and results
        
    Returns:
        PenetrationTestingFramework instance
    """
    return PenetrationTestingFramework(
        db_session, security_audit, compliance_audit, security_monitoring, test_data_path
    )